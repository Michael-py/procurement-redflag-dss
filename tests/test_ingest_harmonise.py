"""Unit tests: ingestion, validation, standardisation, method mapping, supplier resolution."""
from __future__ import annotations

import gzip
import json
from datetime import date

import pandas as pd
import pytest

from conftest import FED, STATE, release
from redflag.config import ConfigError, _validate
from redflag.harmonise import clean_amount, clean_count, map_method, normalise_name, normalise_status, parse_date, resolve_suppliers
from redflag.ingest import ValidationLog, read_source, validate_release


# ----------------------------------------------------------------- ingestion
def test_read_source_counts_and_rejects(tmp_path):
    good = [release(f"ocds-test01-{i}") for i in range(3)]
    lines = [json.dumps(r) for r in good]
    lines.insert(1, "{not json")                                  # invalid JSON
    lines.append(json.dumps({"id": "no-ocid"}))                   # missing ocid
    lines.append(json.dumps(good[0]))                             # duplicate ocid
    lines.append(json.dumps({"ocid": "XYZ-1"}))                   # malformed ocid
    lines.append(json.dumps({"ocid": "ocds-test01-9", "awards": {"id": 1}}))  # awards not a list
    path = tmp_path / "s.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    log = ValidationLog()
    records, stats = read_source(path, "s", log)
    accepted = list(records)
    assert [r["ocid"] for r in accepted] == ["ocds-test01-0", "ocds-test01-1", "ocds-test01-2"]
    assert (stats.lines, stats.accepted, stats.rejected) == (8, 3, 5)
    assert sorted(r["code"] for r in log.rows) == ["awards_not_a_list", "duplicate_ocid", "invalid_json", "malformed_ocid", "missing_ocid"]
    assert len(stats.sha256) == 64


def test_read_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_source(tmp_path / "absent.jsonl.gz", "s", ValidationLog())


@pytest.mark.parametrize("rec,code", [
    ([1, 2], "not_an_object"), ({"ocid": ""}, "missing_ocid"), ({"ocid": 5}, "missing_ocid"),
    ({"ocid": "ocds-a-1", "tender": "x"}, "tender_not_an_object"), ({"ocid": "ocds-a-1"}, None),
])
def test_validate_release(rec, code):
    assert validate_release(rec) == code


# ------------------------------------------------------------------- scalars
@pytest.mark.parametrize("value,expected", [
    ("2022-05-04T11:00:00Z", (date(2022, 5, 4), "ok")),
    ("2022-10-11T20:47:06.000000Z", (date(2022, 10, 11), "ok")),
    ("2022-05-04", (date(2022, 5, 4), "ok")),
    (":00Z", (None, "invalid")), ("2022", (None, "invalid")), ("2022-13-45T00:00:00Z", (None, "invalid")),
    (None, (None, "missing")), ("", (None, "missing")), (20220504, (None, "invalid")),
])
def test_parse_date(value, expected):
    assert parse_date(value) == expected


def test_clean_amount_and_count():
    assert clean_amount(10) == 10.0 and clean_amount(0) is None and clean_amount(-5) is None
    assert clean_amount("10") is None and clean_amount(True) is None and clean_amount(None) is None
    assert clean_count(3) == 3 and clean_count(0) == 0 and clean_count(3.0) == 3
    assert clean_count(-1) is None and clean_count(2.5) is None and clean_count("3") is None


def test_normalise_status():
    allowed = {"active", "complete"}
    assert normalise_status("Complete ", allowed) == "complete"
    assert normalise_status("completed", allowed) == "complete"
    assert normalise_status("ongoing", allowed) == "active"
    assert normalise_status("-- Select Status --", allowed) is None
    assert normalise_status("", allowed) is None


# ------------------------------------------------------------ method mapping
@pytest.mark.parametrize("details,expected", [
    ("National Competitive Bidding", "national_competitive_bidding"),
    ("International Competitive Bidding", "international_competitive_bidding"),
    ("Selective Tendering", "selective_tendering"),
    ("Direct Procurement", "direct_procurement"),
    ("Request for quotation", "request_for_quotation"),
    ("National Shopping", "national_shopping"),
    ("Emergency", "emergency"),
    ("Repeat Procurement", "repeat_procurement"),
    ("Sole Source", "sole_source"),
    ("Direct Labour", "direct_labour"),
])
def test_federal_methods_map_from_details(cfg, details, expected):
    """Every method NOCOPO publishes maps to the controlled vocabulary, even
    though procurementMethod says 'open' for all of them."""
    assert map_method(details, "open", cfg.source(FED), cfg) == (expected, "details")


@pytest.mark.parametrize("details,code,expected", [
    ("In selective procedures, only selected economic operator may submit a tender in response to a contract notice", "selective", ("selective_tendering", "details")),
    ("In direct procedures, only selected economic operator may submit a tender in response to a contract notice", "direct", ("direct_procurement", "details")),
    ("In open procedures, any interested economic operator may submit a tender in response to a contract notice", "open", ("open_tendering", "details")),
    ("Few contractors were invited to bid", "selective", ("selective_tendering", "details")),
    ("Direct labour project", "direct", ("direct_labour", "details")),
    ("direct award to a single contractor", "direct", ("direct_procurement", "details")),
    ("Open Competitive Bidding", "open", ("open_tendering", "details")),
    ("written", "open", ("open_tendering", "code")),
    ("Tender", "selective", ("selective_tendering", "code")),
    (None, "limited", ("selective_tendering", "code")),
    (None, "-- Select Method --", (None, "unmapped")),
    ("", "", (None, "unmapped")),
])
def test_state_methods(cfg, details, code, expected):
    assert map_method(details, code, cfg.source(STATE), cfg) == expected


def test_untrusted_code_is_not_used(cfg):
    """NOCOPO's hard-coded 'open' must not be read as a method."""
    assert map_method(None, "open", cfg.source(FED), cfg) == (None, "unmapped")


def test_method_classes(cfg):
    assert cfg.method_class("national_competitive_bidding") == "competitive_open"
    assert cfg.method_class("selective_tendering") == "competitive_restricted"
    assert cfg.method_class("request_for_quotation") == "quotation"
    assert cfg.method_class("sole_source") == "non_competitive"
    assert cfg.method_class(None) == "unknown"


# -------------------------------------------------------- supplier resolution
def test_normalise_name(cfg):
    legal = set(cfg["suppliers"]["legal_form_tokens"])
    assert normalise_name("Alpha Works Ltd.", legal) == normalise_name("ALPHA WORKS LIMITED", legal) == "ALPHA WORKS"
    assert normalise_name("Messrs  Beta & Sons Nig. Ltd", legal) == "BETA SONS"
    assert normalise_name(None, legal) == ""


def _awards(rows):
    return pd.DataFrame(rows, columns=["award_key", "ocid", "source", "supplier_id_raw", "supplier_name_raw", "supplier_id_blank", "supplier_name_norm"])


def test_resolve_suppliers_paths(cfg):
    awards = _awards([
        ("a1", "o1", FED, "NG-BPP-S1", "Alpha Works Ltd", 0, "ALPHA WORKS"),
        ("a2", "o2", FED, "NG-BPP-", "ALPHA WORKS LIMITED", 1, "ALPHA WORKS"),   # blank id, name matches S1
        ("a3", "o3", FED, "NG-BPP-", "Gamma Supplies", 1, "GAMMA SUPPLIES"),     # blank id, unseen name
        ("a4", "o4", FED, "NG-BPP-", "Gamma Supplies Ltd", 1, "GAMMA SUPPLIES"),
        ("a5", "o5", FED, "NG-BPP-", None, 1, ""),                               # nothing to go on
        ("a6", "o6", FED, "NG-BPP-S7", "Delta", 0, "DELTA"),
        ("a7", "o7", FED, "NG-BPP-S8", "Delta", 0, "DELTA"),
        ("a8", "o8", FED, "NG-BPP-", "Delta", 1, "DELTA"),                       # ambiguous: two ids share the name
        ("a9", "o9", STATE, "NG-CAC-", "Alpha Works Ltd", 1, "ALPHA WORKS"),     # other source: no cross-match
    ])
    log = ValidationLog()
    out, suppliers = resolve_suppliers(awards, cfg, log)
    got = out.set_index("award_key")[["supplier_key", "supplier_resolution"]].to_dict("index")
    assert got["a1"] == {"supplier_key": f"{FED}:NG-BPP-S1", "supplier_resolution": "identifier"}
    assert got["a2"] == {"supplier_key": f"{FED}:NG-BPP-S1", "supplier_resolution": "name_match"}
    assert got["a3"]["supplier_key"] == got["a4"]["supplier_key"] == f"{FED}:NAME:GAMMA SUPPLIES"
    assert got["a3"]["supplier_resolution"] == "name_only"
    assert got["a5"]["supplier_key"] is None and got["a5"]["supplier_resolution"] == "unresolved"
    assert got["a8"] == {"supplier_key": f"{FED}:NAME:DELTA", "supplier_resolution": "name_only"}
    assert got["a9"] == {"supplier_key": f"{STATE}:NAME:ALPHA WORKS", "supplier_resolution": "name_only"}
    assert log.count("supplier_unresolved") == 1
    assert suppliers.set_index("supplier_key").at[f"{FED}:NG-BPP-S1", "n_awards"] == 2


# ------------------------------------------------------------ harmonisation
def test_harmonise_repairs_and_logs(run):
    fed = [
        release("ocds-f-1", award_date="2922-01-01"),                                   # implausible year
        release("ocds-f-2", award_date=":00Z"),                                         # unparseable date
        release("ocds-f-3", amount=0),                                                  # non-positive amount
        release("ocds-f-4", start="2023-04-20", end="2023-03-01"),                      # end before start
        release("ocds-f-5", start="2023-03-01", end="2023-03-15", duration=-14),        # wrong published duration
        release("ocds-f-6", status="Complete "),                                        # status to normalise
        release("ocds-f-7", details=None),                                              # no method details
        release("ocds-f-8", tender=False, award=False),                                 # planning only
    ]
    built = run(fed=fed)
    t = built["tables"]
    award = t["award"].set_index("ocid")
    tender = t["tender"].set_index("ocid")
    log = built["log"]
    assert pd.isna(award.at["ocds-f-1", "award_date"]) and award.at["ocds-f-1", "award_date_raw"].startswith("2922")
    assert log.count("implausible_date") == 1 and log.count("invalid_date") == 1
    assert pd.isna(award.at["ocds-f-3", "amount"]) and log.count("invalid_amount") == 1
    assert pd.isna(tender.at["ocds-f-4", "period_days"]) and log.count("period_end_before_start") == 1
    assert tender.at["ocds-f-5", "period_days"] == 14 and tender.at["ocds-f-5", "period_days_published"] == -14
    assert log.count("period_duration_recomputed") == 1
    assert award.at["ocds-f-6", "status"] == "complete" and log.count("status_normalised") == 1
    assert pd.isna(tender.at["ocds-f-7", "method"]) and tender.at["ocds-f-7", "method_class"] == "unknown"
    assert log.count("method_unmapped") == 1
    assert len(t["contracting_process"]) == 8 and len(built["contracts"]) == 7  # planning-only is stored, not scored


def test_consolidation_keeps_sources_apart(run):
    """TC-05: federal and state records land in one set of tables keyed by ocid."""
    built = run(fed=[release("ocds-f-1"), release("ocds-f-2")],
                state=[release("ocds-s-1", buyer=("works@x.ng", "MINISTRY OF WORKS"), details="open tendering", supplier=("NG-CAC-77", "Zed Ltd"))])
    proc = built["tables"]["contracting_process"]
    assert proc["ocid"].is_unique and set(proc["source"]) == {FED, STATE}
    assert built["log"].count("ocid_clash_across_sources") == 0
    assert set(built["tables"]["buyer"]["buyer_key"]) == {f"{FED}:NG-BPP-B1", f"{STATE}:works@x.ng"}


def test_ocid_clash_across_sources_is_logged(run):
    built = run(fed=[release("ocds-x-1")], state=[release("ocds-x-1", details="open tendering")])
    assert built["tables"]["contracting_process"]["ocid"].is_unique
    assert built["log"].count("ocid_clash_across_sources") == 1


# --------------------------------------------------------------------- config
def test_config_rejects_bad_values(base_cfg):
    import copy

    raw = copy.deepcopy(base_cfg.raw)
    raw["indicators"]["single_bidder"]["weight"] = -1
    with pytest.raises(ConfigError):
        _validate(raw)
    raw = copy.deepcopy(base_cfg.raw)
    raw["methods"]["code_map"]["open"] = "not_a_method"
    with pytest.raises(ConfigError):
        _validate(raw)
    raw = copy.deepcopy(base_cfg.raw)
    del raw["anomaly"]
    with pytest.raises(ConfigError):
        _validate(raw)
