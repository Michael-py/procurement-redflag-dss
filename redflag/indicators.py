"""Localised red-flag indicator engine.

Each indicator is an explicit rule that returns, for every contract, one of
four statuses and a plain-language reason:

    flagged         the rule fired (value in (0, 1] is the flag strength)
    clear           the rule was evaluated and did not fire (value 0)
    not_applicable  the rule does not apply to this kind of contract
    non_computable  the data needed is missing or failed its fitness check

Nothing is given a default: a contract that cannot be assessed says so
(NFR6). The reasons are what the watchlist shows to the analyst (NFR2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INDICATORS, Config

FLAGGED, CLEAR, NOT_APPLICABLE, NON_COMPUTABLE = "flagged", "clear", "not_applicable", "non_computable"

METHOD_LABELS = {
    "international_competitive_bidding": "International Competitive Bidding",
    "national_competitive_bidding": "National Competitive Bidding",
    "open_tendering": "Open Tendering",
    "selective_tendering": "Selective Tendering",
    "request_for_quotation": "Request for Quotation",
    "national_shopping": "National Shopping",
    "direct_procurement": "Direct Procurement",
    "sole_source": "Sole Source",
    "emergency": "Emergency Procurement",
    "repeat_procurement": "Repeat Procurement",
    "direct_labour": "Direct Labour",
}

INDICATOR_LABELS = {
    "single_bidder": "Single bidder in a competitive procedure",
    "short_period": "Short submission period",
    "non_competitive": "Non-competitive method",
    "supplier_concentration": "Buyer-supplier award concentration",
    "award_close_to_estimate": "Award close to estimate",
    "bid_level_screens": "Bid-level collusion screens (R024, R025, R036, R038)",
    "anomaly": "Unusual contract profile (anomaly score)",
}


def method_label(method) -> str:
    return METHOD_LABELS.get(method, "an unmapped method") if isinstance(method, str) else "an unmapped method"


def _result(index: pd.Index, name: str) -> pd.DataFrame:
    return pd.DataFrame({"indicator": name, "status": NON_COMPUTABLE, "value": np.nan, "reason": ""}, index=index)


def _set(res: pd.DataFrame, mask, status: str, reason, value=np.nan) -> None:
    """Apply an outcome to rows not already decided. Rules are written from
    most to least blocking, so the first matching rule wins."""
    mask = pd.Series(mask, index=res.index).fillna(False).astype(bool) & (res["reason"] == "")
    res.loc[mask, "status"] = status
    res.loc[mask, "value"] = value[mask] if isinstance(value, pd.Series) else value
    if isinstance(reason, pd.Series):
        reason = reason[mask]
        if reason.isna().any():  # an explanation must never be blank
            raise RuntimeError(f"{res['indicator'].iat[0]}: rule produced an empty reason")
    res.loc[mask, "reason"] = reason


# ----------------------------------------------------------------- rules
def single_bidder(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    res = _result(c.index, "single_bidder")
    applies = set(cfg["indicators"]["single_bidder"]["applies_to_classes"])
    labels = c["method"].map(method_label)
    n = c["num_tenderers"]
    _set(res, c["has_tender"] == 0, NON_COMPUTABLE, "No tender section published")
    _set(res, c["method_class"] == "unknown", NON_COMPUTABLE, "Procurement method could not be mapped")
    _set(res, ~c["method_class"].isin(applies), NOT_APPLICABLE, "A single tenderer is expected under " + labels)
    _set(res, n.isna(), NON_COMPUTABLE, "Number of tenderers not published")
    _set(res, n == 0, NON_COMPUTABLE, "Zero tenderers reported for an awarded tender")
    _set(res, n == 1, FLAGGED, "Single bidder in " + labels, 1.0)
    _set(res, n > 1, CLEAR, n.astype("string") + " tenderers in " + labels, 0.0)
    return res


def short_period(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    res = _result(c.index, "short_period")
    p = cfg["indicators"]["short_period"]
    minimum = c["method_class"].map(p["minimum_days"])
    severe = c["method_class"].map(p["severe_days"])
    days = c["period_days"].astype("Float64")
    labels = c["method"].map(method_label)
    d_txt = c["period_days"].astype("string")
    min_txt = minimum.astype("Int64").astype("string")
    _set(res, c["has_tender"] == 0, NON_COMPUTABLE, "No tender section published")
    _set(res, c["method_class"] == "unknown", NON_COMPUTABLE, "Procurement method could not be mapped")
    _set(res, minimum.isna(), NOT_APPLICABLE, "No minimum submission period applies to " + labels)
    _set(res, days.isna(), NON_COMPUTABLE, "Tender-period dates missing or invalid")
    _set(res, days == 0, NON_COMPUTABLE, "Tender period starts and ends at the same moment (placeholder dates)")
    _set(res, days > p["max_plausible_days"], NON_COMPUTABLE, "Tender period of " + d_txt + " days is implausible")
    _set(res, days < severe, FLAGGED,
         "Submission period of " + d_txt + " days for " + labels + " (minimum " + min_txt + ")", 1.0)
    _set(res, days < minimum, FLAGGED,
         "Submission period of " + d_txt + " days for " + labels + " (minimum " + min_txt + ")", float(p["moderate_score"]))
    _set(res, days >= minimum, CLEAR, "Submission period of " + d_txt + " days meets the minimum of " + min_txt, 0.0)
    return res


def non_competitive(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    res = _result(c.index, "non_competitive")
    labels = c["method"].map(method_label)
    is_nc = c["method_class"] == "non_competitive"
    known = c["method_class"] != "unknown"
    share = (is_nc & known).groupby(c["buyer_key"]).transform("sum") / known.groupby(c["buyer_key"]).transform("sum")
    share_txt = (share * 100).round(0).astype("Int64").astype("string")
    _set(res, c["has_tender"] == 0, NON_COMPUTABLE, "No tender section published")
    _set(res, ~known, NON_COMPUTABLE, "Procurement method could not be mapped")
    with_share = "Awarded by " + labels + " (non-competitive; " + share_txt + "% of this buyer's tenders are non-competitive)"
    _set(res, is_nc, FLAGGED, with_share.fillna("Awarded by " + labels + " (non-competitive)"), 1.0)
    _set(res, ~is_nc, CLEAR, labels + " is a competitive method", 0.0)
    return res


def supplier_concentration(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    res = _result(c.index, "supplier_concentration")
    p = cfg["indicators"]["supplier_concentration"]
    usable = c["supplier_key"].notna() & c["award_amount"].notna() & c["buyer_key"].notna()
    u = c[usable]
    buyer_value = u.groupby("buyer_key")["award_amount"].transform("sum")
    buyer_awards = u.groupby("buyer_key")["ocid"].transform("count")
    buyer_suppliers = u.groupby("buyer_key")["supplier_key"].transform("nunique")
    pair_value = u.groupby(["buyer_key", "supplier_key"])["award_amount"].transform("sum")
    pair_awards = u.groupby(["buyer_key", "supplier_key"])["ocid"].transform("count")
    share = (pair_value / buyer_value).reindex(c.index)
    buyer_awards = buyer_awards.reindex(c.index)
    buyer_suppliers = buyer_suppliers.reindex(c.index)
    pair_awards = pair_awards.reindex(c.index)

    pct = (share * 100).round(0).astype("Int64").astype("string")
    pa = pair_awards.astype("Int64").astype("string")
    ba = buyer_awards.astype("Int64").astype("string")
    _set(res, c["n_live_awards"] == 0, NON_COMPUTABLE, "No award published")
    _set(res, c["buyer_key"].isna(), NON_COMPUTABLE, "Buyer not identified")
    _set(res, c["supplier_key"].isna(), NON_COMPUTABLE, "Supplier could not be resolved (blank identifier and no usable name)")
    _set(res, c["award_amount"].isna(), NON_COMPUTABLE, "Award value missing or not positive")
    small = (buyer_awards < p["min_buyer_awards"]) | (buyer_suppliers < p["min_buyer_suppliers"])
    _set(res, small, NOT_APPLICABLE, "Buyer has too few awards or suppliers (" + ba + " awards) for a share to be meaningful")
    hit = (share >= p["share_threshold"]) & (pair_awards >= p["min_pair_awards"])
    _set(res, hit, FLAGGED, "Supplier won " + pct + "% of this buyer's awarded value across " + pa + " awards (buyer total: " + ba + " awards)", 1.0)
    _set(res, ~hit, CLEAR, "Supplier holds " + pct + "% of this buyer's awarded value", 0.0)
    return res


def estimate_fitness(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-source fitness of the tender estimate: the share of award-estimate
    pairs in which the estimate is an exact copy of the award value."""
    pairs = c[c["estimate_amount"].notna() & c["award_amount"].notna()]
    out = pairs.groupby("source").agg(pairs=("ocid", "count"), copied=("estimate_copied", "sum"))
    out = out.reindex(sorted(c["source"].unique())).fillna(0).astype(int)
    out["copied_share"] = (out["copied"] / out["pairs"]).where(out["pairs"] > 0)
    out["fit"] = (out["pairs"] > 0) & (out["copied_share"] < cfg["indicators"]["award_close_to_estimate"]["copied_share_unfit"])
    return out


def award_close_to_estimate(c: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    res = _result(c.index, "award_close_to_estimate")
    p = cfg["indicators"]["award_close_to_estimate"]
    fit = estimate_fitness(c, cfg)
    unfit_share = c["source"].map(fit["copied_share"].where(~fit["fit"] & (fit["pairs"] > 0)))
    est, amt = c["estimate_amount"], c["award_amount"]
    diff = ((amt - est).abs() / est).astype(float)
    diff_txt = (diff * 100).round(2).astype("string")
    _set(res, unfit_share.notna(), NON_COMPUTABLE,
         "Estimate field unfit for this source: it equals the award value in "
         + (unfit_share * 100).round(1).astype("string") + "% of pairs (data-quality flag)")
    _set(res, est.isna(), NON_COMPUTABLE, "No tender estimate published")
    _set(res, amt.isna(), NON_COMPUTABLE, "No award value published")
    _set(res, c["estimate_copied"] == 1, NON_COMPUTABLE, "Estimate equals the award value exactly, so it is not independent (data-quality flag)")
    _set(res, diff <= p["tolerance"], FLAGGED, "Award within " + diff_txt + "% of the tender estimate", 1.0)
    _set(res, diff > p["tolerance"], CLEAR, "Award differs from the estimate by " + diff_txt + "%", 0.0)
    return res


def bid_level_screens(c: pd.DataFrame, cfg: Config, bids: pd.DataFrame | None = None) -> pd.DataFrame:
    res = _result(c.index, "bid_level_screens")
    has_bids = c["ocid"].isin(set(bids["ocid"])) if bids is not None and len(bids) else pd.Series(False, index=c.index)
    _set(res, ~has_bids, NON_COMPUTABLE, "No bid-level data published by this source")
    _set(res, has_bids, NON_COMPUTABLE, "Bid data present but bid-level screens are not enabled in this release")
    return res


RULES = {
    "single_bidder": single_bidder,
    "short_period": short_period,
    "non_competitive": non_competitive,
    "supplier_concentration": supplier_concentration,
    "award_close_to_estimate": award_close_to_estimate,
    "bid_level_screens": bid_level_screens,
}


def compute_indicators(contracts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run every rule. Returns a long table: ocid, indicator, status, value, reason."""
    c = contracts.reset_index(drop=True)
    parts = []
    for name in INDICATORS:
        res = RULES[name](c, cfg)
        res.insert(0, "ocid", c["ocid"].values)
        parts.append(res)
    out = pd.concat(parts, ignore_index=True)
    undecided = out["reason"] == ""
    if undecided.any():  # a rule left a contract without an outcome: fail loudly
        raise RuntimeError(f"indicator rules left {int(undecided.sum())} results undecided")
    return out


def summarise(results: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """Counts of each status per source and indicator."""
    merged = results.merge(contracts[["ocid", "source"]], on="ocid")
    table = merged.pivot_table(index=["source", "indicator"], columns="status", values="ocid", aggfunc="count", fill_value=0)
    for col in (FLAGGED, CLEAR, NOT_APPLICABLE, NON_COMPUTABLE):
        if col not in table:
            table[col] = 0
    table = table[[FLAGGED, CLEAR, NOT_APPLICABLE, NON_COMPUTABLE]]
    table["contracts"] = table.sum(axis=1)
    evaluated = table[FLAGGED] + table[CLEAR]
    table["flag_rate"] = (table[FLAGGED] / evaluated).where(evaluated > 0)
    table["fitness"] = 1 - table[NON_COMPUTABLE] / table["contracts"]
    return table.reset_index()
