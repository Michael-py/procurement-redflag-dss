"""Optional unsupervised anomaly layer (isolation forest).

Kept separate from the rule engine so that the core output stays fully
explainable. The model is fitted per source, because a state contract is not
anomalous merely for looking unlike a federal one. With a fixed seed and a
fixed row order the scores are reproducible (NFR4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config

MIN_ROWS = 100
METHOD_CLASSES = ["competitive_open", "competitive_restricted", "quotation", "non_competitive"]


def build_features(c: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=c.index)
    f["log_amount"] = np.log10(c["award_amount"].astype(float).where(c["award_amount"] > 0))
    f["num_tenderers"] = c["num_tenderers"].astype(float)
    f["period_days"] = c["period_days"].astype(float).clip(upper=365)
    f["n_awards"] = c["n_live_awards"].astype(float)
    buyer_n = c.groupby("buyer_key")["ocid"].transform("count").astype(float)
    f["log_buyer_contracts"] = np.log10(buyer_n.where(buyer_n > 0))
    pair_n = c.groupby(["buyer_key", "supplier_key"])["ocid"].transform("count").astype(float)
    f["pair_share"] = pair_n / buyer_n
    numeric = list(f.columns)
    for col in numeric:  # missingness is itself informative, so keep it as a feature
        f[col + "_missing"] = f[col].isna().astype(float)
        f[col] = f[col].fillna(f[col].median() if f[col].notna().any() else 0.0)
    for cls in METHOD_CLASSES:
        f["method_" + cls] = (c["method_class"] == cls).astype(float)
    return f


def anomaly_scores(contracts: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return a score in [0, 1] per contract (1 = most unusual within its
    source), or NaN where the layer is disabled or the source is too small."""
    scores = pd.Series(np.nan, index=contracts.index, name="anomaly_score")
    p = cfg["anomaly"]
    if not p["enabled"]:
        return scores
    for _, group in contracts.sort_values("ocid").groupby("source", sort=True):
        if len(group) < MIN_ROWS:
            continue
        x = build_features(group)
        model = IsolationForest(n_estimators=p["n_estimators"], random_state=p["random_state"], n_jobs=1)
        raw = -model.fit(x.values).score_samples(x.values)
        # percentile rank within the source: comparable across sources and bounded
        scores.loc[group.index] = pd.Series(raw, index=group.index).rank(method="average", pct=True).round(6)
    return scores
