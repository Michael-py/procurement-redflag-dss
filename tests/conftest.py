"""Shared fixtures: a small, hand-built corpus with known expected outcomes."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redflag.config import Config, load_config  # noqa: E402
from redflag.pipeline import build  # noqa: E402

FED = "fed_test"
STATE = "state_test"


def release(ocid, *, buyer=("NG-BPP-B1", "MINISTRY ONE"), details="National Competitive Bidding", code="open",
            tenderers=4, start="2023-03-01", end="2023-04-20", duration=None, estimate=None, amount=1_000_000.0,
            supplier=("NG-BPP-S1", "Alpha Works Ltd"), award_date="2023-05-10", status="active", tender=True, award=True):
    rec = {"ocid": ocid, "id": ocid + "-r1", "date": "2023-06-01T00:00:00Z", "tag": ["compiled"],
           "buyer": {"id": buyer[0], "name": buyer[1]}}
    if tender:
        t = {"id": "t-" + ocid, "title": "Title " + ocid, "status": "complete", "procurementMethod": code}
        if details is not None:
            t["procurementMethodDetails"] = details
        if tenderers is not None:
            t["numberOfTenderers"] = tenderers
        if start is not None or end is not None:
            t["tenderPeriod"] = {}
            if start is not None:
                t["tenderPeriod"]["startDate"] = start + "T00:00:00Z" if len(start) == 10 else start
            if end is not None:
                t["tenderPeriod"]["endDate"] = end + "T00:00:00Z" if len(end) == 10 else end
            if duration is not None:
                t["tenderPeriod"]["durationInDays"] = duration
        if estimate is not None:
            t["value"] = {"amount": estimate, "currency": "NGN"}
        rec["tender"] = t
    if award:
        rec["awards"] = [{"id": "a-" + ocid, "date": award_date + "T00:00:00Z" if award_date and len(award_date) == 10 else award_date,
                          "status": status, "value": {"amount": amount, "currency": "NGN"},
                          "suppliers": [{"id": supplier[0], "name": supplier[1]}]}]
    return rec


@pytest.fixture(scope="session")
def base_cfg() -> Config:
    return load_config(ROOT / "config.toml")


@pytest.fixture()
def cfg(base_cfg) -> Config:
    """Real thresholds and weights, two small test sources."""
    raw = copy.deepcopy(base_cfg.raw)
    raw["sources"] = [
        {"id": FED, "label": "Federal test", "level": "federal", "file": "fed.jsonl", "registry_publication": 0,
         "method_code_trusted": False, "blank_id_patterns": ["^NG-BPP-$"]},
        {"id": STATE, "label": "State test", "level": "state", "file": "state.jsonl", "registry_publication": 0,
         "method_code_trusted": True, "blank_id_patterns": ["^NG-CAC-$", "^NG-TIN-$"]},
    ]
    return Config(raw=raw, path=base_cfg.path, sha256=base_cfg.sha256, root=base_cfg.root)


@pytest.fixture()
def run(cfg):
    """Build the pipeline in memory from lists of releases."""

    def _run(fed=(), state=()):
        return build(cfg, extra_releases={FED: list(fed), STATE: list(state)})

    return _run


def outcome(built, ocid, indicator):
    r = built["results"]
    row = r[(r["ocid"] == ocid) & (r["indicator"] == indicator)]
    assert len(row) == 1, f"expected one result for {ocid}/{indicator}"
    return row.iloc[0]
