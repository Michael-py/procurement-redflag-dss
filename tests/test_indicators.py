"""Indicator correctness: each rule against fixtures with known expected outcomes."""
from __future__ import annotations

import pytest

from conftest import outcome, release


def _status(built, ocid, indicator):
    return outcome(built, ocid, indicator)["status"]


# -------------------------------------------------------------- single bidder
def test_single_bidder_flagged_with_evidence(run):
    """TC-03: a competitive tender with one bidder is flagged and says why."""
    built = run(fed=[release("ocds-f-1", tenderers=1)])
    r = outcome(built, "ocds-f-1", "single_bidder")
    assert r["status"] == "flagged" and r["value"] == 1.0
    assert r["reason"] == "Single bidder in National Competitive Bidding"


@pytest.mark.parametrize("details,tenderers,expected", [
    ("National Competitive Bidding", 2, "clear"),
    ("International Competitive Bidding", 1, "flagged"),
    ("Selective Tendering", 1, "flagged"),
    ("Direct Procurement", 1, "not_applicable"),
    ("Sole Source", 1, "not_applicable"),
    ("Emergency", 1, "not_applicable"),
    ("Repeat Procurement", 1, "not_applicable"),
    ("Request for quotation", 1, "not_applicable"),
    ("National Competitive Bidding", None, "non_computable"),
    ("National Competitive Bidding", 0, "non_computable"),
    (None, 1, "non_computable"),
])
def test_single_bidder_by_method(run, details, tenderers, expected):
    built = run(fed=[release("ocds-f-1", details=details, tenderers=tenderers)])
    assert _status(built, "ocds-f-1", "single_bidder") == expected


def test_direct_procurement_under_open_code_is_not_single_bidder(run):
    """TC-07: procurementMethod 'open' with details 'Direct Procurement' maps
    to non-competitive, so the single-bidder flag is not applied."""
    built = run(fed=[release("ocds-f-1", code="open", details="Direct Procurement", tenderers=1)])
    tender = built["tables"]["tender"].iloc[0]
    assert (tender["method"], tender["method_class"]) == ("direct_procurement", "non_competitive")
    assert _status(built, "ocds-f-1", "single_bidder") == "not_applicable"
    assert _status(built, "ocds-f-1", "non_competitive") == "flagged"


# --------------------------------------------------------------- short period
@pytest.mark.parametrize("details,days,status,value", [
    ("National Competitive Bidding", 42, "clear", 0.0),
    ("National Competitive Bidding", 41, "flagged", 0.5),
    ("National Competitive Bidding", 14, "flagged", 0.5),
    ("National Competitive Bidding", 13, "flagged", 1.0),
    ("Selective Tendering", 14, "clear", 0.0),
    ("Selective Tendering", 13, "flagged", 0.5),
    ("Selective Tendering", 6, "flagged", 1.0),
    ("Request for quotation", 7, "clear", 0.0),
    ("Request for quotation", 2, "flagged", 1.0),
    ("Direct Procurement", 1, "not_applicable", None),
    ("National Competitive Bidding", 0, "non_computable", None),
    ("National Competitive Bidding", 400, "non_computable", None),
])
def test_short_period_thresholds(run, details, days, status, value):
    from datetime import date, timedelta

    end = (date(2023, 3, 1) + timedelta(days=days)).isoformat()
    built = run(fed=[release("ocds-f-1", details=details, start="2023-03-01", end=end)])
    r = outcome(built, "ocds-f-1", "short_period")
    assert r["status"] == status
    if value is not None:
        assert r["value"] == value


def test_short_period_missing_or_reversed_dates(run):
    built = run(fed=[release("ocds-f-1", start=None, end=None), release("ocds-f-2", start="2023-04-20", end="2023-03-01")])
    assert _status(built, "ocds-f-1", "short_period") == "non_computable"
    assert _status(built, "ocds-f-2", "short_period") == "non_computable"


def test_short_period_uses_dates_not_published_duration(run):
    """A wrong durationInDays (Ebonyi publishes negatives) must not drive the flag."""
    built = run(fed=[release("ocds-f-1", start="2023-03-01", end="2023-04-20", duration=-50)])
    assert _status(built, "ocds-f-1", "short_period") == "clear"


# ------------------------------------------------------------ non-competitive
@pytest.mark.parametrize("details,expected", [
    ("Direct Procurement", "flagged"), ("Sole Source", "flagged"), ("Emergency", "flagged"),
    ("Repeat Procurement", "flagged"), ("Direct Labour", "flagged"),
    ("National Competitive Bidding", "clear"), ("Selective Tendering", "clear"), ("National Shopping", "clear"),
    (None, "non_computable"),
])
def test_non_competitive(run, details, expected):
    built = run(fed=[release("ocds-f-1", details=details)])
    assert _status(built, "ocds-f-1", "non_competitive") == expected


def test_non_competitive_reason_reports_buyer_share(run):
    fed = [release("ocds-f-1", details="Direct Procurement")] + [release(f"ocds-f-{i}") for i in range(2, 5)]
    built = run(fed=fed)
    assert "25% of this buyer's tenders are non-competitive" in outcome(built, "ocds-f-1", "non_competitive")["reason"]


# ------------------------------------------------------- supplier concentration
def _buyer_with_awards(shares):
    """One buyer; `shares` is a list of (supplier_no, amount)."""
    return [release(f"ocds-f-{i}", supplier=(f"NG-BPP-S{s}", f"Supplier {s} Ltd"), amount=amount)
            for i, (s, amount) in enumerate(shares, start=1)]


def test_concentration_flagged(run):
    # 10 awards, supplier 1 holds 2 awards worth 40% of the value
    shares = [(1, 200), (1, 200)] + [(s, 75) for s in range(2, 10)]
    built = run(fed=_buyer_with_awards(shares))
    r = outcome(built, "ocds-f-1", "supplier_concentration")
    assert r["status"] == "flagged"
    assert r["reason"] == "Supplier won 40% of this buyer's awarded value across 2 awards (buyer total: 10 awards)"
    assert _status(built, "ocds-f-3", "supplier_concentration") == "clear"


def test_concentration_needs_repeat_awards(run):
    # one very large single award must not be flagged as concentration
    shares = [(1, 5000)] + [(s, 75) for s in range(2, 11)]
    built = run(fed=_buyer_with_awards(shares))
    assert _status(built, "ocds-f-1", "supplier_concentration") == "clear"


def test_concentration_small_buyer_not_applicable(run):
    built = run(fed=_buyer_with_awards([(1, 100), (1, 100), (2, 10)]))
    assert _status(built, "ocds-f-1", "supplier_concentration") == "not_applicable"


def test_blank_supplier_id_resolved_by_name_or_marked_unresolved(run):
    """TC-08: an award with identifier 'NG-BPP-' is resolved by normalised
    name, or left unresolved and excluded from the concentration indicator."""
    fed = [
        release("ocds-f-1", supplier=("NG-BPP-S1", "Alpha Works Ltd")),
        release("ocds-f-2", supplier=("NG-BPP-", "ALPHA WORKS LIMITED")),
        release("ocds-f-3", supplier=("NG-BPP-", "")),
    ]
    built = run(fed=fed)
    award = built["tables"]["award"].set_index("ocid")
    assert award.at["ocds-f-2", "supplier_key"] == award.at["ocds-f-1", "supplier_key"]
    assert award.at["ocds-f-2", "supplier_resolution"] == "name_match"
    assert award.at["ocds-f-3", "supplier_resolution"] == "unresolved"
    r = outcome(built, "ocds-f-3", "supplier_concentration")
    assert r["status"] == "non_computable" and "could not be resolved" in r["reason"]


def test_blank_ids_do_not_form_one_giant_supplier(run):
    """Ten different unnamed-identifier suppliers must not be merged."""
    fed = [release(f"ocds-f-{i}", supplier=("NG-BPP-", f"Vendor {chr(64 + i)} Ltd")) for i in range(1, 11)]
    built = run(fed=fed)
    assert built["tables"]["award"]["supplier_key"].nunique() == 10
    assert all(_status(built, f"ocds-f-{i}", "supplier_concentration") == "clear" for i in range(1, 11))


# ---------------------------------------------------------- close to estimate
def test_estimate_equal_to_award_is_non_computable(run):
    """TC-06: an estimate equal to the award value is a copy, not a signal."""
    built = run(fed=[release("ocds-f-1", estimate=1_000_000.0, amount=1_000_000.0)])
    r = outcome(built, "ocds-f-1", "award_close_to_estimate")
    assert r["status"] == "non_computable" and "data-quality flag" in r["reason"]
    assert built["tables"]["award"].iloc[0]["estimate_copied"] == 1


def test_source_with_copied_estimates_is_unfit_as_a_whole(run):
    # 10 copies and 1 genuine close estimate: 91% copied, so the field is unfit
    fed = [release(f"ocds-f-{i}", estimate=1000.0 + i, amount=1000.0 + i) for i in range(1, 11)]
    fed.append(release("ocds-f-11", estimate=1000.0, amount=995.0))
    built = run(fed=fed)
    r = outcome(built, "ocds-f-11", "award_close_to_estimate")
    assert r["status"] == "non_computable" and "unfit for this source" in r["reason"]


@pytest.mark.parametrize("estimate,amount,expected", [
    (1000.0, 995.0, "flagged"), (1000.0, 1010.0, "flagged"), (1000.0, 1011.0, "clear"),
    (1000.0, 800.0, "clear"), (None, 800.0, "non_computable"),
])
def test_close_to_estimate_with_independent_estimates(run, estimate, amount, expected):
    # a state source with genuinely different estimates, as in Plateau
    state = [release(f"ocds-s-{i}", details="open tendering", estimate=2000.0, amount=1500.0) for i in range(2, 6)]
    state.append(release("ocds-s-1", details="open tendering", estimate=estimate, amount=amount))
    built = run(state=state)
    assert _status(built, "ocds-s-1", "award_close_to_estimate") == expected


# ------------------------------------------------------------------ bid level
def test_bid_level_screens_always_non_computable(run):
    """TC-02: with no bid values, collusion screens are reported as non-computable."""
    built = run(fed=[release("ocds-f-1"), release("ocds-f-2", tenderers=1)])
    for ocid in ("ocds-f-1", "ocds-f-2"):
        r = outcome(built, ocid, "bid_level_screens")
        assert r["status"] == "non_computable" and r["reason"] == "No bid-level data published by this source"


# ------------------------------------------------------------------- general
def test_every_contract_gets_every_indicator_with_a_reason(run):
    fed = [release("ocds-f-1"), release("ocds-f-2", tender=False), release("ocds-f-3", award=False), release("ocds-f-4", details=None, tenderers=None)]
    built = run(fed=fed)
    r = built["results"]
    assert len(r) == 4 * 6
    assert (r["reason"].str.len() > 0).all()
    assert set(r["status"]) <= {"flagged", "clear", "not_applicable", "non_computable"}
    assert r[r["status"].isin(["not_applicable", "non_computable"])]["value"].isna().all()
