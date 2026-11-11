"""End-to-end pipeline: ingest -> harmonise -> score quality -> indicators ->
anomaly -> prioritise -> warehouse and exports.

Run with:  python -m redflag run
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import DISCLAIMER, __version__
from .anomaly import anomaly_scores
from .config import Config, load_config
from .harmonise import Harmoniser
from .indicators import compute_indicators, summarise
from .ingest import ValidationLog, read_source
from .prioritise import score_contracts, watchlist
from .quality import RawProfiler, comparison_table, warehouse_metrics
from .warehouse import contracts_frame, write_warehouse

log = logging.getLogger("redflag")


def build(cfg: Config, extra_releases: dict[str, list[dict]] | None = None) -> dict:
    """Run every stage in memory and return the resulting tables.

    `extra_releases` lets tests and the synthetic evaluation inject releases
    for a configured source without touching files on disk.
    """
    timings: dict[str, float] = {}
    vlog = ValidationLog()
    harmoniser = Harmoniser(cfg, vlog)
    profiler = RawProfiler(cfg)
    source_rows = []

    t0 = time.perf_counter()
    for src in cfg.sources:
        if extra_releases is not None and src["id"] in extra_releases:
            records, stats = _from_memory(extra_releases[src["id"]], src["id"], vlog)
        else:
            records, stats = read_source(cfg.resolve("raw_dir") / src["file"], src["id"], vlog)
        for rec in records:
            profiler.add(rec, src)
            harmoniser.add(rec, src)
        source_rows.append({"label": src["label"], "level": src["level"], **asdict(stats)})
        log.info("ingested %-24s lines=%d accepted=%d rejected=%d", src["id"], stats.lines, stats.accepted, stats.rejected)
    timings["ingest_and_flatten"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tables = harmoniser.frames()
    clashes = tables["contracting_process"]["ocid"].duplicated(keep=False)
    if clashes.any():  # the same ocid published by two sources: keep the first, log the rest
        dup = tables["contracting_process"][tables["contracting_process"]["ocid"].duplicated(keep="first")]
        for row in dup.itertuples():
            vlog.add(row.source, row.ocid, "error", "ocid_clash_across_sources")
        keep = ~tables["contracting_process"]["ocid"].duplicated(keep="first")
        kept = set(zip(tables["contracting_process"][keep]["ocid"], tables["contracting_process"][keep]["source"]))
        tables["contracting_process"] = tables["contracting_process"][keep]
        tables["tender"] = tables["tender"].drop_duplicates("ocid", keep="first")
        tables["award"] = tables["award"][[(o, s) in kept for o, s in zip(tables["award"]["ocid"], tables["award"]["source"])]]
    timings["resolve_and_frame"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dq = pd.DataFrame(profiler.metrics() + warehouse_metrics(tables, cfg))
    timings["quality"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    contracts = contracts_frame(tables)
    results = compute_indicators(contracts, cfg)
    timings["indicators"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    anomaly = anomaly_scores(contracts, cfg)
    timings["anomaly"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = score_contracts(contracts, results, anomaly if cfg["anomaly"]["enabled"] else None, cfg)
    timings["prioritise"] = time.perf_counter() - t0

    tables.update(
        source_file=pd.DataFrame(source_rows)[["source", "label", "level", "file", "sha256", "bytes", "lines", "accepted", "rejected"]],
        validation_log=pd.DataFrame(vlog.rows, columns=["source", "ocid", "line_no", "severity", "code", "detail"]),
        dq_metric=dq,
        indicator_result=results,
        contract_score=scores,
    )
    return {"tables": tables, "contracts": contracts, "results": results, "scores": scores, "dq": dq,
            "anomaly": anomaly, "timings": timings, "log": vlog}


def _from_memory(releases: list[dict], source_id: str, vlog: ValidationLog):
    from .ingest import IngestStats, validate_release

    stats = IngestStats(source=source_id, file="<memory>", sha256="-", bytes=0)

    def gen():
        seen = set()
        for i, rec in enumerate(releases, start=1):
            stats.lines += 1
            code = validate_release(rec) or ("duplicate_ocid" if rec["ocid"] in seen else None)
            if code:
                stats.rejected += 1
                vlog.add(source_id, rec.get("ocid") if isinstance(rec, dict) else None, "error", code, "", i)
                continue
            seen.add(rec["ocid"])
            stats.accepted += 1
            yield rec

    return gen(), stats


def run(config_path: str | Path | None = None) -> dict:
    """Run the pipeline and persist the warehouse and CSV exports."""
    cfg = load_config(config_path)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    built = build(cfg)
    tables = built["tables"]

    manifest = {
        "system_version": __version__,
        "config_sha256": cfg.sha256,
        "retrieval_date": cfg["run"]["retrieval_date"],
        "disclaimer": DISCLAIMER,
    }
    t1 = time.perf_counter()
    write_warehouse(cfg.resolve("warehouse"), tables, manifest, marts={"contract_mart": built["contracts"]})
    built["timings"]["write_warehouse"] = time.perf_counter() - t1

    out = cfg.resolve("output_dir")
    out.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    wl = watchlist(built["contracts"], built["scores"], cfg["prioritisation"]["watchlist_size"])
    wl.to_csv(out / "watchlist.csv", index=False, encoding="utf-8-sig")
    comparison_table(built["dq"]).to_csv(out / "data_quality_raw_vs_warehouse.csv", index=False)
    summarise(built["results"], built["contracts"]).to_csv(out / "indicator_summary.csv", index=False)
    codes = tables["validation_log"].groupby(["source", "severity", "code"]).size().rename("count").reset_index()
    codes.to_csv(out / "validation_log_summary.csv", index=False)
    built["timings"]["exports"] = time.perf_counter() - t1
    built["timings"]["total"] = time.perf_counter() - t0

    # Run-time facts go to a log file, not into the warehouse, so that the
    # warehouse itself is byte-for-byte reproducible from inputs and config.
    with open(out / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{started.isoformat(timespec='seconds')} version={__version__} config={cfg.sha256[:12]} "
                + " ".join(f"{k}={v:.2f}s" for k, v in built["timings"].items())
                + f" processes={len(tables['contracting_process'])} contracts={len(built['contracts'])}\n")
    log.info("done in %.1fs: %d processes, %d scored contracts", built["timings"]["total"],
             len(tables["contracting_process"]), len(built["contracts"]))
    built["cfg"] = cfg
    return built
