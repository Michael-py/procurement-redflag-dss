"""Quality metrics, prioritisation, anomaly layer and end-to-end integration."""
from __future__ import annotations

import gzip
import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from conftest import FED, STATE, release
from redflag import pipeline
from redflag.config import Config
from redflag.evaluate import precision_at_k, ranking_digest, seeded_precision
from redflag.prioritise import score_contracts
from redflag.synthetic import generate


def _corpus():
    fed = []
    for i in range(1, 13):  # twelve ordinary federal contracts across four suppliers
        fed.append(release(f"ocds-f-{i:02d}", supplier=(f"NG-BPP-S{i % 4}", f"Supplier {i % 4} Ltd"), amount=1_000_000.0 + i,
                           estimate=1_000_000.0 + i))
    fed.append(release("ocds-f-20", tenderers=1, start="2023-03-01", end="2023-03-06", supplier=("NG-BPP-S9", "Nine Ltd"), amount=9e6, estimate=9e6))
    fed.append(release("ocds-f-21", details="Direct Procurement", tenderers=1, supplier=("NG-BPP-", ""), amount=5e6, estimate=5e6))
    fed.append(release("ocds-f-22", tender=False, award=False))
    state = [release("ocds-s-01", buyer=("NG-TIN-", "MINISTRY OF WORKS"), details="selective tendering", code="selective", tenderers=3,
                     supplier=("NG-CAC-1", "Zed Ltd"), status="Complete", duration=-50)]
    return fed, state


# ------------------------------------------------------------ quality metrics
def test_quality_metrics_raw_vs_warehouse(run):
    fed, state = _corpus()
    built = run(fed=fed, state=state)
    dq = built["dq"].set_index(["source", "stage", "dimension", "metric"])

    def v(source, stage, dim, metric):
        return dq.loc[(source, stage, dim, metric), ["numerator", "denominator"]].tolist()

    # the federal method code is a constant, so it distinguishes nothing until mapped
    assert v(FED, "raw", "fitness", "method_distinguishes_competition") == [0, 14]
    assert v(FED, "warehouse", "fitness", "method_distinguishes_competition") == [14, 14]
    # 1 of 14 supplier identifiers is blank and has no name, so it stays unresolved
    assert v(FED, "raw", "fitness", "supplier_identity_usable") == [13, 14]
    assert v(FED, "warehouse", "fitness", "supplier_identity_usable") == [13, 14]
    # every federal estimate is a copy; harmonisation detects this but cannot repair it
    assert v(FED, "raw", "fitness", "estimate_independent_of_award") == [0, 14]
    assert v(FED, "warehouse", "fitness", "estimate_independent_of_award") == [0, 14]
    # the state's blank buyer id is replaced by a name-based key, its status is normalised
    assert v(STATE, "raw", "fitness", "buyer_identifier_usable") == [0, 1]
    assert v(STATE, "warehouse", "fitness", "buyer_identifier_usable") == [1, 1]
    assert v(STATE, "raw", "validity", "award_status") == [0, 1]
    assert v(STATE, "warehouse", "validity", "award_status") == [1, 1]
    assert v(STATE, "raw", "consistency", "tender_period_duration") == [0, 1]
    assert ((built["dq"]["value"].dropna() >= 0) & (built["dq"]["value"].dropna() <= 1)).all()


def test_suspected_default_tenderer_count(run):
    state = [release(f"ocds-s-{i}", details="selective tendering", tenderers=3) for i in range(8)]
    state += [release(f"ocds-s-9{i}", details="selective tendering", tenderers=n) for i, n in enumerate([1, 2])]
    built = run(state=state)
    row = built["dq"].query("source == @STATE and stage == 'warehouse' and metric == 'tenderers_not_defaulted'").iloc[0]
    assert (row["numerator"], row["denominator"]) == (2, 10)
    assert "exactly 3 tenderers" in row["note"]


# -------------------------------------------------------------- prioritisation
def test_score_is_the_documented_weighted_sum(run, cfg):
    fed, state = _corpus()
    built = run(fed=fed, state=state)
    scores = built["scores"].set_index("ocid")
    w = cfg.weights
    total = sum(w.values())
    # single bidder (1.0) + severe short period (1.0); the anomaly layer is off for sources under 100 rows
    assert scores.at["ocds-f-20", "risk_score"] == pytest.approx((w["single_bidder"] + w["short_period"]) / total, abs=1e-6)
    assert scores.at["ocds-f-20", "n_flags"] == 2
    assert scores.at["ocds-f-20", "rank"] == 1
    assert scores.at["ocds-f-20", "explanation"] == (
        "Single bidder in National Competitive Bidding; Submission period of 5 days for National Competitive Bidding (minimum 42)")
    assert scores.at["ocds-f-21", "risk_score"] == pytest.approx(w["non_competitive"] / total, abs=1e-6)
    assert "Supplier could not be resolved" in scores.at["ocds-f-21", "not_assessed"]
    assert scores["risk_score"].between(0, 1).all()
    assert scores["rank"].tolist() == sorted(scores["rank"]) or scores["rank"].is_unique


def test_missing_data_does_not_raise_the_score(run):
    """A contract with nothing computable must rank below one with a real flag."""
    built = run(fed=[release("ocds-f-1", details=None, tenderers=None, start=None, end=None, supplier=("NG-BPP-", "")),
                     release("ocds-f-2", tenderers=1)])
    s = built["scores"].set_index("ocid")
    assert s.at["ocds-f-1", "risk_score"] == 0 and s.at["ocds-f-1", "coverage"] == 0
    assert s.at["ocds-f-2", "rank"] < s.at["ocds-f-1", "rank"]


def test_ranking_is_deterministic(run):
    """TC-04: identical inputs and configuration give an identical ranking."""
    fed, state = _corpus()
    a, b = run(fed=fed, state=state), run(fed=list(reversed(fed)), state=state)
    assert ranking_digest(a["scores"]) == ranking_digest(b["scores"])
    pd.testing.assert_frame_equal(a["scores"], b["scores"])


def test_weights_are_configurable(run, cfg):
    """NFR5: changing weights re-ranks without touching code."""
    fed, state = _corpus()
    built = run(fed=fed, state=state)
    weights = {k: 0.0 for k in cfg.weights}
    weights["non_competitive"] = 1.0
    rescored = score_contracts(built["contracts"], built["results"], None, cfg, weights).set_index("ocid")
    assert rescored.at["ocds-f-21", "rank"] == 1 and rescored.at["ocds-f-21", "risk_score"] == 1.0
    assert rescored.at["ocds-f-20", "risk_score"] == 0.0
    with pytest.raises(ValueError):
        score_contracts(built["contracts"], built["results"], None, cfg, {k: 0.0 for k in cfg.weights})


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, [1, 2, 4]) == {1: 1.0, 2: 0.5, 4: 0.5}


# --------------------------------------------------------------- anomaly layer
def test_anomaly_scores_are_bounded_and_reproducible(cfg):
    from redflag.pipeline import build
    from redflag.synthetic import SOURCE_ID, synthetic_config

    releases, _ = generate(n_background=280, n_seeded=20)
    scfg = synthetic_config(cfg)
    a = build(scfg, extra_releases={SOURCE_ID: releases})["anomaly"]
    b = build(scfg, extra_releases={SOURCE_ID: releases})["anomaly"]
    assert a.notna().all() and a.between(0, 1).all()
    pd.testing.assert_series_equal(a, b)


def test_anomaly_layer_can_be_switched_off(cfg, run):
    cfg.raw["anomaly"]["enabled"] = False
    built = run(fed=[release("ocds-f-1")])
    assert built["scores"]["anomaly_score"].isna().all()


def test_seeded_cases_rank_first(cfg):
    table, info = seeded_precision(cfg, ks=(25, 100), n_random=50)
    system = table.set_index("strategy").loc["System (rules + anomaly)"]
    assert system["p@25"] == 1.0 and system["p@100"] == 1.0
    assert info["seeded_all_in_top_n"]


def test_synthetic_generator_is_reproducible_and_labelled():
    a, seeded_a = generate()
    b, seeded_b = generate()
    assert a == b and seeded_a == seeded_b and len(seeded_a) == 100
    assert all(r["ocid"].startswith("ocds-synth0-") and r["buyer"]["name"].startswith("SYNTHETIC") for r in a)


# ----------------------------------------------------------------- integration
@pytest.fixture()
def disk_cfg(cfg, tmp_path) -> Config:
    fed, state = _corpus()
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for name, records in (("fed.jsonl", fed), ("state.jsonl", state)):
        with gzip.open(raw_dir / (name + ".gz"), "wt", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in records) + "\n")
    cfg.raw["sources"][0]["file"] = "fed.jsonl.gz"
    cfg.raw["sources"][1]["file"] = "state.jsonl.gz"
    cfg.raw["paths"] = {"raw_dir": "data/raw", "warehouse": "data/warehouse.sqlite", "output_dir": "outputs",
                        "audit_findings": "data/reference/none.csv"}
    return Config(raw=cfg.raw, path=cfg.path, sha256=cfg.sha256, root=tmp_path)


def test_end_to_end_run_writes_warehouse_and_exports(disk_cfg, monkeypatch):
    """TC-01 and TC-05: files in, one consistent warehouse and the exports out."""
    monkeypatch.setattr(pipeline, "load_config", lambda _path=None: disk_cfg)
    pipeline.run()
    db = disk_cfg.resolve("warehouse")
    with closing(sqlite3.connect(db)) as con:
        count = lambda table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: E731
        assert count("contracting_process") == 16           # every accepted line became one process
        assert con.execute("SELECT SUM(accepted), SUM(rejected) FROM source_file").fetchone() == (16, 0)
        assert count("tender") == 15 and count("award") == 15
        assert count("bid") == 0                             # table exists, no source fills it
        assert count("indicator_result") == 15 * 6
        assert count("contract_score") == 15
        assert con.execute("SELECT COUNT(DISTINCT ocid) FROM contracting_process").fetchone()[0] == 16
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert con.execute("SELECT ocid FROM contract_score WHERE rank = 1").fetchone()[0] == "ocds-f-20"
        keys = dict(con.execute("SELECT key, value FROM run_manifest"))
        assert keys["config_sha256"] == disk_cfg.sha256 and "not findings" in keys["disclaimer"]
    out = disk_cfg.resolve("output_dir")
    wl = pd.read_csv(out / "watchlist.csv")
    assert wl["rank"].tolist() == list(range(1, 16)) and wl.iloc[0]["ocid"] == "ocds-f-20"
    for name in ("data_quality_raw_vs_warehouse.csv", "indicator_summary.csv", "validation_log_summary.csv", "run_log.txt"):
        assert (out / name).exists()


def test_rerun_reproduces_the_warehouse_content(disk_cfg, monkeypatch):
    """NFR3: a second run from the same inputs gives the same stored results."""
    monkeypatch.setattr(pipeline, "load_config", lambda _path=None: disk_cfg)

    def snapshot():
        pipeline.run()
        with closing(sqlite3.connect(disk_cfg.resolve("warehouse"))) as con:
            return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
                    for t in ("contracting_process", "tender", "award", "indicator_result", "contract_score", "dq_metric")}

    assert snapshot() == snapshot()


def test_failed_run_leaves_previous_warehouse_intact(disk_cfg, monkeypatch):
    monkeypatch.setattr(pipeline, "load_config", lambda _path=None: disk_cfg)
    pipeline.run()
    db = disk_cfg.resolve("warehouse")
    before = db.read_bytes()
    (disk_cfg.resolve("raw_dir") / "state.jsonl.gz").unlink()   # a source goes missing
    with pytest.raises(FileNotFoundError):
        pipeline.run()
    assert db.read_bytes() == before
