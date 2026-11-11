"""System-level evaluation on the four measures of Section 3.10.

    1. data-quality improvement   raw published data vs the warehouse
    2. indicator correctness      the pytest suite (run separately)
    3. prioritisation quality     seeded synthetic cases and watchlist make-up
    4. external plausibility      buyer-level flag rates vs audit findings

Everything is written to outputs/evaluation/ so that the figures quoted in
the report can be regenerated with `python -m redflag evaluate`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import INDICATORS, Config, load_config
from .indicators import CLEAR, FLAGGED, NOT_APPLICABLE, summarise
from .pipeline import build
from .prioritise import score_contracts
from .quality import comparison_table
from .synthetic import SOURCE_ID, generate, synthetic_config

log = logging.getLogger("redflag.evaluate")


# ------------------------------------------------- 1. data-quality improvement
def naive_vs_harmonised(contracts: pd.DataFrame, results: pd.DataFrame, awards: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """What a direct application of each rule to the published fields would
    flag, against what the harmonised engine flags.

    'Naive' means using the OCDS fields at face value: procurementMethod as
    the method, durationInDays as the period, the supplier identifier as the
    supplier and tender.value as an independent estimate.
    """
    c = contracts.set_index("ocid")
    status = results.pivot(index="ocid", columns="indicator", values="status").reindex(c.index)
    code = c["method_code_raw"].fillna("").str.strip().str.lower()
    code_competitive = code.isin(["open", "selective", "limited"])
    rows = []

    def add(indicator, naive, note):
        system = status[indicator] == FLAGGED
        for source in sorted(c["source"].unique()):
            m = c["source"] == source
            rows.append({
                "source": source, "indicator": indicator,
                "naive_flagged": int((naive & m).sum()), "system_flagged": int((system & m).sum()),
                "naive_only": int((naive & ~system & m).sum()), "system_only": int((system & ~naive & m).sum()),
                "note": note,
            })

    add("single_bidder", code_competitive & (c["num_tenderers"] == 1),
        "naive: procurementMethod in open/selective/limited and one tenderer")
    p = cfg["indicators"]["short_period"]
    published = c["period_days_published"].astype("Float64")
    add("short_period", (code.eq("open") & (published < p["minimum_days"]["competitive_open"])).fillna(False),
        "naive: procurementMethod 'open' and published durationInDays below the open-bidding minimum")
    add("non_competitive", code.eq("direct"), "naive: procurementMethod equals 'direct'")
    diff = (c["award_amount"] - c["estimate_amount"]).abs() / c["estimate_amount"]
    add("award_close_to_estimate", (diff <= cfg["indicators"]["award_close_to_estimate"]["tolerance"]).fillna(False),
        "naive: award within tolerance of tender.value, including exact copies")

    # concentration on raw identifiers: a blank identifier behaves like one giant supplier
    sc = cfg["indicators"]["supplier_concentration"]
    a = c[c["award_amount"].notna() & c["buyer_key"].notna()].copy()
    main = awards.sort_values(["ocid", "amount"], ascending=[True, False]).drop_duplicates("ocid").set_index("ocid")
    a["raw_id"] = main["supplier_id_raw"].reindex(a.index)
    a = a[a["raw_id"].notna()]
    bv = a.groupby("buyer_key")["award_amount"].transform("sum")
    bn = a.groupby("buyer_key")["award_amount"].transform("count")
    bs = a.groupby("buyer_key")["raw_id"].transform("nunique")
    pv = a.groupby(["buyer_key", "raw_id"])["award_amount"].transform("sum")
    pn = a.groupby(["buyer_key", "raw_id"])["award_amount"].transform("count")
    hit = (pv / bv >= sc["share_threshold"]) & (pn >= sc["min_pair_awards"]) & (bn >= sc["min_buyer_awards"]) & (bs >= sc["min_buyer_suppliers"])
    add("supplier_concentration", hit.reindex(c.index).fillna(False),
        "naive: suppliers grouped by published identifier, blank identifiers included")
    return pd.DataFrame(rows)


def dq_summary(dq: pd.DataFrame) -> pd.DataFrame:
    """Raw vs warehouse per dimension, pooled over sources (sum of numerators
    over sum of denominators, so large and small sources are weighted by size)."""
    g = dq.groupby(["dimension", "metric", "stage"])[["numerator", "denominator"]].sum()
    g["value"] = g["numerator"] / g["denominator"].where(g["denominator"] > 0)
    wide = g["value"].unstack("stage").reset_index()
    wide["change"] = wide["warehouse"] - wide["raw"]
    return wide


# ------------------------------------------------- 3. prioritisation quality
def precision_at_k(ranked_ocids: list[str], positives: set[str], ks: list[int]) -> dict[int, float]:
    return {k: sum(1 for o in ranked_ocids[:k] if o in positives) / k for k in ks}


def seeded_precision(cfg: Config, ks=(25, 50, 100, 200), n_random: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Mechanics check on a synthetic corpus with known seeded cases."""
    releases, seeded = generate()
    built = build(synthetic_config(cfg), extra_releases={SOURCE_ID: releases})
    contracts, scores = built["contracts"], built["scores"]
    ks = [k for k in ks if k <= len(contracts)]
    system = scores.sort_values("rank")["ocid"].tolist()
    weights = dict(cfg.weights)
    weights["anomaly"] = 0.0
    rules_only = score_contracts(contracts, built["results"], None, cfg, weights).sort_values("rank")["ocid"].tolist()
    largest = contracts.sort_values(["award_amount", "ocid"], ascending=[False, True])["ocid"].tolist()

    rng = random.Random(cfg["anomaly"]["random_state"])
    pool = sorted(contracts["ocid"])
    rand = {k: [] for k in ks}
    for _ in range(n_random):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        for k, v in precision_at_k(shuffled, seeded, ks).items():
            rand[k].append(v)

    rows = []
    for name, ranking in (("System (rules + anomaly)", system), ("System (rules only)", rules_only), ("Largest contract first", largest)):
        rows.append({"strategy": name, **{f"p@{k}": v for k, v in precision_at_k(ranking, seeded, ks).items()}})
    rows.append({"strategy": f"Random selection (mean of {n_random} draws)", **{f"p@{k}": float(np.mean(rand[k])) for k in ks}})
    info = {
        "contracts": len(contracts), "seeded": len(seeded), "base_rate": len(seeded) / len(contracts),
        "indicator_summary": summarise(built["results"], contracts).to_dict("records"),
        "seeded_all_in_top_n": all(o in seeded for o in system[: len(seeded)]),
    }
    return pd.DataFrame(rows), info


def watchlist_composition(contracts: pd.DataFrame, scores: pd.DataFrame, results: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """What an analyst reviewing k contracts would see under each strategy,
    on the real data. This describes the lists; it does not claim accuracy."""
    c = contracts.merge(scores[["ocid", "rank", "risk_score", "n_flags"]], on="ocid")
    single = set(results[(results["indicator"] == "single_bidder") & (results["status"] == FLAGGED)]["ocid"])
    c["single_bidder"] = c["ocid"].isin(single)
    rng = np.random.default_rng(seed)
    strategies = {
        "System watchlist": c.sort_values("rank").head(k),
        "Largest contract first": c.sort_values(["award_amount", "ocid"], ascending=[False, True]).head(k),
        "Random selection": c.sort_values("ocid").iloc[rng.permutation(len(c))[:k]],
    }
    rows = []
    for name, sel in strategies.items():
        rows.append({
            "strategy": name, "contracts": len(sel),
            "mean_flags": sel["n_flags"].mean(),
            "share_with_2plus_flags": (sel["n_flags"] >= 2).mean(),
            "share_with_no_flag": (sel["n_flags"] == 0).mean(),
            "share_single_bidder": sel["single_bidder"].mean(),
            "distinct_buyers": sel["buyer_key"].nunique(),
            "total_value_ngn_bn": sel["award_amount"].sum() / 1e9,
            "median_value_ngn_m": sel["award_amount"].median() / 1e6,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------- 4. external plausibility
def external_plausibility(contracts: pd.DataFrame, scores: pd.DataFrame, cfg: Config, min_contracts: int = 5) -> tuple[pd.DataFrame | None, list[dict]]:
    """Spearman rank correlation between each matched federal MDA's flag rate
    and the number of procurement-related issues the Auditor-General reported
    for it. Returns (per-MDA table, one result row per time window)."""
    path = cfg.resolve("audit_findings")
    if not path.exists():
        return None, []
    audit = pd.read_csv(path)
    audit = audit[audit["buyer_key"].notna() & (audit["buyer_key"] != "")]
    audit = audit.groupby("buyer_key", as_index=False).agg(
        mda=("mda", "first"), issues_total=("issues_total", "sum"), issues_procurement=("issues_procurement", "sum"))

    c = contracts.merge(scores[["ocid", "risk_score", "n_flags"]], on="ocid")
    c = c[c["source"] == "nigeria_portal"].copy()
    c["year"] = pd.to_datetime(c["award_date"], errors="coerce").dt.year
    windows = {"2024": [2024], "2023-2024": [2023, 2024], "2022-2024": [2022, 2023, 2024], "all years": None}
    results, table = [], None
    for name, years in windows.items():
        sel = c if years is None else c[c["year"].isin(years)]
        per = sel.groupby("buyer_key").agg(contracts=("ocid", "count"), flagged=("n_flags", lambda s: int((s > 0).sum())),
                                           mean_risk=("risk_score", "mean")).reset_index()
        per["flag_rate"] = per["flagged"] / per["contracts"]
        m = audit.merge(per, on="buyer_key", how="inner")
        m = m[m["contracts"] >= min_contracts]
        row = {"window": name, "mdas_matched": len(m), "min_contracts": min_contracts}
        for measure in ("flag_rate", "mean_risk"):
            for target in ("issues_procurement", "issues_total"):
                if len(m) >= 5 and m[measure].nunique() > 1 and m[target].nunique() > 1:
                    rho, pval = stats.spearmanr(m[measure], m[target])
                    row[f"rho_{measure}_vs_{target}"] = float(rho)
                    row[f"p_{measure}_vs_{target}"] = float(pval)
                else:
                    row[f"rho_{measure}_vs_{target}"] = None
                    row[f"p_{measure}_vs_{target}"] = None
        results.append(row)
        if name == "all years":
            table = m.sort_values("flag_rate", ascending=False)
    return table, results


# ------------------------------------------------------------------- driver
def ranking_digest(scores: pd.DataFrame) -> str:
    payload = "\n".join(f"{r.rank}|{r.ocid}|{r.risk_score:.6f}" for r in scores.sort_values("rank").itertuples())
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate(config_path: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    out = cfg.resolve("output_dir") / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config_sha256": cfg.sha256, "retrieval_date": cfg["run"]["retrieval_date"]}

    log.info("building warehouse tables (run 1 of 2)")
    built = build(cfg)
    tables, contracts, results, scores = built["tables"], built["contracts"], built["results"], built["scores"]

    # 1. data quality
    comparison_table(built["dq"]).to_csv(out / "dq_by_source.csv", index=False)
    pooled = dq_summary(built["dq"])
    pooled.to_csv(out / "dq_pooled.csv", index=False)
    naive = naive_vs_harmonised(contracts, results, tables["award"], cfg)
    naive.to_csv(out / "naive_vs_harmonised.csv", index=False)
    ind = summarise(results, contracts)
    ind.to_csv(out / "indicator_summary.csv", index=False)
    vl = tables["validation_log"]
    vl.groupby(["source", "severity", "code"]).size().rename("count").reset_index().to_csv(out / "validation_log_summary.csv", index=False)
    summary["sources"] = tables["source_file"].to_dict("records")
    summary["counts"] = {
        "processes": len(tables["contracting_process"]), "tenders": len(tables["tender"]), "awards": len(tables["award"]),
        "buyers": len(tables["buyer"]), "suppliers": len(tables["supplier"]), "scored_contracts": len(contracts),
        "validation_log_rows": len(vl),
    }
    summary["supplier_resolution"] = tables["award"].groupby(["source", "supplier_resolution"]).size().rename("n").reset_index().to_dict("records")
    summary["method_mapping"] = (tables["tender"].merge(tables["contracting_process"][["ocid", "source"]], on="ocid")
                                 .groupby(["source", "method_origin"]).size().rename("n").reset_index().to_dict("records"))

    # 3a. determinism: an independent second run must give the same ranking
    log.info("building warehouse tables (run 2 of 2, determinism check)")
    second = build(cfg)
    d1, d2 = ranking_digest(scores), ranking_digest(second["scores"])
    summary["determinism"] = {"run1": d1, "run2": d2, "identical": d1 == d2,
                              "indicator_results_identical": bool(results.equals(second["results"]))}

    # 3b. seeded precision and composition
    log.info("seeded synthetic precision check")
    prec, info = seeded_precision(cfg)
    prec.to_csv(out / "seeded_precision.csv", index=False)
    summary["seeded"] = {k: v for k, v in info.items() if k != "indicator_summary"}
    pd.DataFrame(info["indicator_summary"]).to_csv(out / "synthetic_indicator_summary.csv", index=False)
    k = cfg["prioritisation"]["watchlist_size"]
    comp = watchlist_composition(contracts, scores, results, k, cfg["anomaly"]["random_state"])
    comp.to_csv(out / "watchlist_composition.csv", index=False)

    # 4. external plausibility
    table, ext = external_plausibility(contracts, scores, cfg)
    if table is None:
        summary["external_plausibility"] = "audit findings file not found"
    else:
        table.to_csv(out / "external_plausibility_mdas.csv", index=False)
        pd.DataFrame(ext).to_csv(out / "external_plausibility.csv", index=False)
        summary["external_plausibility"] = ext

    summary["timings_seconds"] = {k: round(v, 2) for k, v in built["timings"].items()}
    summary["score_distribution"] = {
        "flag_counts": {int(k): int(v) for k, v in scores["n_flags"].value_counts().sort_index().items()},
        "max_score": float(scores["risk_score"].max()),
        "watchlist_min_score": float(scores.sort_values("rank").head(k)["risk_score"].min()),
    }
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("evaluation written to %s", out)
    return summary
