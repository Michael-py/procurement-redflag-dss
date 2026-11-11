"""Explainable, deterministic prioritisation.

    risk_score = sum_i(weight_i * flag_i) [+ w_anomaly * anomaly_score]
                 / sum of all weights

The denominator is the total configured weight, not the weight of the
indicators that happened to be computable. A contract therefore cannot reach
the top of the list just because most of its indicators are missing; how much
of the model could be assessed is reported separately as `coverage`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INDICATORS, Config
from .indicators import CLEAR, FLAGGED, INDICATOR_LABELS, NON_COMPUTABLE

ANOMALY_NOTE_PERCENTILE = 0.95


def score_contracts(
    contracts: pd.DataFrame,
    results: pd.DataFrame,
    anomaly: pd.Series | None,
    cfg: Config,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine indicator results into a ranked table, one row per contract."""
    weights = dict(weights if weights is not None else cfg.weights)
    rule_w = pd.Series({k: float(weights.get(k, 0.0)) for k in INDICATORS})
    w_anom = float(weights.get("anomaly", 0.0)) if anomaly is not None else 0.0
    total = rule_w.sum() + w_anom
    if total <= 0:
        raise ValueError("total weight must be positive")

    order = contracts["ocid"]
    value = results.pivot(index="ocid", columns="indicator", values="value").reindex(index=order, columns=INDICATORS)
    status = results.pivot(index="ocid", columns="indicator", values="status").reindex(index=order, columns=INDICATORS)
    reason = results.pivot(index="ocid", columns="indicator", values="reason").reindex(index=order, columns=INDICATORS)

    flagged = status.eq(FLAGGED)
    weighted = value.where(flagged, 0.0).fillna(0.0).mul(rule_w, axis=1)
    anom = pd.Series(np.nan, index=order) if anomaly is None else pd.Series(anomaly.values, index=order)
    raw = weighted.sum(axis=1) + w_anom * anom.fillna(0.0)

    assessed = status.isin([FLAGGED, CLEAR]).mul(rule_w, axis=1).sum(axis=1)
    scored_w = rule_w[rule_w > 0]

    out = pd.DataFrame(index=order)
    out["risk_score"] = (raw / total).round(6)
    out["n_flags"] = flagged.loc[:, scored_w.index].sum(axis=1).astype(int)
    out["coverage"] = (assessed / scored_w.sum()).round(4)
    out["anomaly_score"] = anom

    # Plain-language explanation: triggered flags in descending contribution.
    contrib_order = weighted.columns[np.argsort(-rule_w.values, kind="stable")]
    explanations, gaps = [], []
    for ocid in order:
        parts = [reason.at[ocid, k] for k in contrib_order if flagged.at[ocid, k] and rule_w[k] > 0]
        a = anom.at[ocid]
        if w_anom > 0 and pd.notna(a) and a >= ANOMALY_NOTE_PERCENTILE:
            parts.append(f"Unusual profile for this source (anomaly percentile {a:.2f})")
        explanations.append("; ".join(parts) if parts else "No flags triggered")
        missing = [f"{INDICATOR_LABELS[k]}: {reason.at[ocid, k]}" for k in INDICATORS if status.at[ocid, k] == NON_COMPUTABLE and rule_w[k] > 0]
        gaps.append("; ".join(missing))
    out["explanation"] = explanations
    out["not_assessed"] = gaps

    out = out.reset_index()
    amount = contracts.set_index("ocid")["award_amount"].reindex(out["ocid"]).fillna(-1).values
    out["_amount"] = amount
    # Fully specified ordering: identical inputs always give an identical ranking.
    out = out.sort_values(["risk_score", "n_flags", "_amount", "ocid"], ascending=[False, False, False, True], kind="mergesort")
    out["rank"] = np.arange(1, len(out) + 1)
    return out.drop(columns="_amount").reset_index(drop=True)[
        ["ocid", "rank", "risk_score", "n_flags", "coverage", "anomaly_score", "explanation", "not_assessed"]
    ]


def watchlist(contracts: pd.DataFrame, scores: pd.DataFrame, size: int | None = None) -> pd.DataFrame:
    """Join scores to the descriptive columns an analyst needs."""
    cols = ["ocid", "source", "buyer_name", "title", "method", "num_tenderers", "period_days",
            "award_amount", "award_date", "supplier_name"]
    wl = scores.merge(contracts[cols], on="ocid", how="left").sort_values("rank")
    return wl.head(size) if size else wl
