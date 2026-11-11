"""Reproducible data-quality metrics, measured twice.

The same questions are asked of the data as published ('raw') and of the
harmonised warehouse, so the improvement the pipeline delivers is measured
rather than asserted. Every metric is a numerator over a stated denominator.

Dimensions (Batini & Scannapieco, 2016):
  completeness  a value is present
  validity      a present value is well-formed and plausible
  consistency   values agree with each other
  fitness       the field can actually support the indicator that needs it
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

from .config import Config
from .harmonise import clean_amount, clean_count, is_blank_id, parse_date

OCDS_METHOD_CODES = {"open", "selective", "limited", "direct"}
TENDER_STATUSES = {"planning", "planned", "active", "cancelled", "unsuccessful", "complete", "withdrawn"}
AWARD_STATUSES = {"pending", "active", "cancelled", "unsuccessful", "complete"}


class RawProfiler:
    """Accumulates quality counts from releases exactly as published."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_date = date.fromisoformat(cfg["run"]["min_plausible_date"])
        self.max_date = date.fromisoformat(cfg["run"]["retrieval_date"])
        self.n: dict[str, Counter] = defaultdict(Counter)
        self.method_codes: dict[str, Counter] = defaultdict(Counter)
        self.tenderer_values: dict[str, Counter] = defaultdict(Counter)

    def _plausible(self, value) -> bool:
        d, status = parse_date(value)
        return status == "ok" and self.min_date <= d <= self.max_date

    def add(self, rec: dict, source_cfg: dict) -> None:
        n = self.n[source_cfg["id"]]
        n["processes"] += 1
        buyer = rec.get("buyer") or {}
        n["buyer_id_present"] += bool(isinstance(buyer.get("id"), str) and buyer["id"].strip())
        n["buyer_id_usable"] += not is_blank_id(buyer.get("id"), source_cfg)

        tender = rec.get("tender") or {}
        if tender:
            n["tenders"] += 1
            code = tender.get("procurementMethod")
            code = code.strip().lower() if isinstance(code, str) else None
            details = tender.get("procurementMethodDetails")
            n["method_present"] += bool(code) or bool(isinstance(details, str) and details.strip())
            n["method_code_valid"] += code in OCDS_METHOD_CODES
            self.method_codes[source_cfg["id"]][code] += 1
            nt = tender.get("numberOfTenderers")
            n["tenderers_present"] += nt is not None
            n["tenderers_valid"] += clean_count(nt) is not None and nt > 0
            if clean_count(nt) is not None:
                self.tenderer_values[source_cfg["id"]][int(nt)] += 1
            period = tender.get("tenderPeriod") or {}
            start, s_ok = parse_date(period.get("startDate"))
            end, e_ok = parse_date(period.get("endDate"))
            if s_ok != "missing" and e_ok != "missing":
                n["period_present"] += 1
                valid = s_ok == "ok" and e_ok == "ok" and self._plausible(period.get("startDate")) and self._plausible(period.get("endDate"))
                span = (end - start).days if valid else None
                n["period_valid"] += bool(valid and 0 < span <= 365)
                published = period.get("durationInDays")
                if valid and isinstance(published, (int, float)):
                    n["duration_published"] += 1
                    n["duration_consistent"] += int(published) == span
            status = tender.get("status")
            if isinstance(status, str) and status.strip():
                n["tender_status_present"] += 1
                n["tender_status_valid"] += status in TENDER_STATUSES
            n["estimate_present"] += clean_amount((tender.get("value") or {}).get("amount")) is not None

        estimate = clean_amount((tender.get("value") or {}).get("amount")) if tender else None
        for award in rec.get("awards") or []:
            if not isinstance(award, dict):
                continue
            n["awards"] += 1
            raw_date = award.get("date")
            if raw_date not in (None, ""):
                n["award_date_present"] += 1
                n["award_date_valid"] += self._plausible(raw_date)
            amount = (award.get("value") or {}).get("amount")
            n["award_amount_present"] += amount is not None
            n["award_amount_valid"] += clean_amount(amount) is not None
            n["award_currency_present"] += bool((award.get("value") or {}).get("currency"))
            status = award.get("status")
            if isinstance(status, str) and status.strip():
                n["award_status_present"] += 1
                n["award_status_valid"] += status in AWARD_STATUSES
            suppliers = award.get("suppliers") or []
            sup = suppliers[0] if suppliers and isinstance(suppliers[0], dict) else {}
            n["supplier_id_present"] += bool(isinstance(sup.get("id"), str) and sup["id"].strip())
            n["supplier_id_usable"] += not is_blank_id(sup.get("id"), source_cfg)
            n["supplier_name_present"] += bool(isinstance(sup.get("name"), str) and sup["name"].strip())
            if estimate is not None and clean_amount(amount) is not None:
                n["estimate_pairs"] += 1
                n["estimate_independent"] += estimate != amount

    def metrics(self) -> list[dict]:
        rows = []
        for source, n in self.n.items():
            codes = self.method_codes[source]
            informative = len([c for c in codes if c in OCDS_METHOD_CODES]) > 1
            m = lambda dim, metric, num, den, note="": rows.append(_row(source, "raw", dim, metric, num, den, note))  # noqa: E731
            m("completeness", "buyer_identifier", n["buyer_id_present"], n["processes"])
            m("completeness", "procurement_method", n["method_present"], n["tenders"])
            m("completeness", "number_of_tenderers", n["tenderers_present"], n["tenders"])
            m("completeness", "tender_period", n["period_present"], n["tenders"])
            m("completeness", "tender_estimate", n["estimate_present"], n["tenders"])
            m("completeness", "award_date", n["award_date_present"], n["awards"])
            m("completeness", "award_value", n["award_amount_present"], n["awards"])
            m("completeness", "supplier_identifier", n["supplier_id_present"], n["awards"])
            m("completeness", "supplier_name", n["supplier_name_present"], n["awards"])
            m("validity", "award_date", n["award_date_valid"], n["award_date_present"])
            m("validity", "award_value", n["award_amount_valid"], n["award_amount_present"])
            m("validity", "tender_period", n["period_valid"], n["period_present"])
            m("validity", "tender_status", n["tender_status_valid"], n["tender_status_present"])
            m("validity", "award_status", n["award_status_valid"], n["award_status_present"])
            m("validity", "number_of_tenderers", n["tenderers_valid"], n["tenderers_present"])
            m("consistency", "tender_period_duration", n["duration_consistent"], n["duration_published"],
              "published durationInDays equals end minus start")
            m("fitness", "method_distinguishes_competition", n["method_code_valid"] if informative else 0, n["tenders"],
              "" if informative else "procurementMethod carries a single value for the whole source")
            m("fitness", "buyer_identifier_usable", n["buyer_id_usable"], n["processes"])
            m("fitness", "supplier_identity_usable", n["supplier_id_usable"], n["awards"])
            m("fitness", "estimate_independent_of_award", n["estimate_independent"], n["estimate_pairs"])
            rows.append(_tenderer_default_row(source, "raw", self.tenderer_values[source], self.cfg))
        return rows


def _row(source, stage, dimension, metric, num, den, note="") -> dict:
    num, den = int(num), int(den)
    return {"source": source, "stage": stage, "dimension": dimension, "metric": metric,
            "numerator": num, "denominator": den, "value": round(num / den, 6) if den else None, "note": note}


def _tenderer_default_row(source: str, stage: str, values: Counter, cfg: Config) -> dict:
    """Share of tenders NOT sitting on a suspected default value. A single
    value above 1 that dominates a source suggests a form default rather than
    a real count (Osun and Ebonyi report exactly three tenderers very often)."""
    total = sum(values.values())
    above_one = {v: k for v, k in values.items() if v > 1}
    if not total or not above_one:
        return _row(source, stage, "fitness", "tenderers_not_defaulted", total, total, "")
    top_value, top_count = max(above_one.items(), key=lambda kv: (kv[1], -kv[0]))
    share = top_count / total
    if share >= cfg["fitness"]["default_value_share"]:
        return _row(source, stage, "fitness", "tenderers_not_defaulted", total - top_count, total,
                    f"suspected default: {top_count} of {total} tenders report exactly {top_value} tenderers")
    return _row(source, stage, "fitness", "tenderers_not_defaulted", total, total, "")


def warehouse_metrics(tables: dict[str, pd.DataFrame], cfg: Config) -> list[dict]:
    """The same metrics, measured on the harmonised tables."""
    rows = []
    proc, tender, award = tables["contracting_process"], tables["tender"], tables["award"]
    tender = tender.merge(proc[["ocid", "source"]], on="ocid")
    published_duration = tables.get("_published_duration")
    for source in sorted(proc["source"].unique()):
        p, t, a = proc[proc["source"] == source], tender[tender["source"] == source], award[award["source"] == source]
        m = lambda dim, metric, num, den, note="": rows.append(_row(source, "warehouse", dim, metric, num, den, note))  # noqa: E731
        m("completeness", "buyer_identifier", p["buyer_key"].notna().sum(), len(p))
        m("completeness", "procurement_method", t["method"].notna().sum(), len(t))
        m("completeness", "number_of_tenderers", t["num_tenderers"].notna().sum(), len(t))
        has_period = t["period_start"].notna() & t["period_end"].notna()
        m("completeness", "tender_period", has_period.sum(), len(t))
        m("completeness", "tender_estimate", t["estimate_amount"].notna().sum(), len(t))
        m("completeness", "award_date", a["award_date"].notna().sum(), len(a))
        m("completeness", "award_value", a["amount"].notna().sum(), len(a))
        m("completeness", "supplier_identifier", a["supplier_key"].notna().sum(), len(a))
        m("completeness", "supplier_name", a["supplier_name_raw"].notna().sum(), len(a))
        # Invalid values are nulled during harmonisation, so what remains is valid.
        m("validity", "award_date", a["award_date"].notna().sum(), a["award_date"].notna().sum())
        m("validity", "award_value", a["amount"].notna().sum(), a["amount"].notna().sum())
        days = t["period_days"].astype("Float64")
        m("validity", "tender_period", ((days > 0) & (days <= 365)).sum(), has_period.sum())
        m("validity", "tender_status", t["status"].notna().sum(), t["status"].notna().sum())
        m("validity", "award_status", a["status"].notna().sum(), a["status"].notna().sum())
        nt = t["num_tenderers"].astype("Float64")
        m("validity", "number_of_tenderers", (nt > 0).sum(), nt.notna().sum())
        m("consistency", "tender_period_duration", days.notna().sum(), days.notna().sum(),
          "duration is recomputed from the dates")
        m("fitness", "method_distinguishes_competition", (t["method_class"] != "unknown").sum(), len(t))
        m("fitness", "buyer_identifier_usable", p["buyer_key"].notna().sum(), len(p))
        m("fitness", "supplier_identity_usable", a["supplier_key"].notna().sum(), len(a))
        pairs = a.merge(t[["ocid", "estimate_amount"]], on="ocid")
        pairs = pairs[pairs["estimate_amount"].notna() & pairs["amount"].notna()]
        m("fitness", "estimate_independent_of_award", (pairs["estimate_copied"] == 0).sum(), len(pairs),
          "unchanged by harmonisation: a copied estimate cannot be repaired, only detected")
        rows.append(_tenderer_default_row(source, "warehouse", Counter(t["num_tenderers"].dropna().astype(int)), cfg))
    return rows


def comparison_table(dq: pd.DataFrame) -> pd.DataFrame:
    """Raw vs warehouse, one row per source and metric."""
    wide = dq.pivot_table(index=["source", "dimension", "metric"], columns="stage", values="value", aggfunc="first").reset_index()
    for col in ("raw", "warehouse"):
        if col not in wide:
            wide[col] = None
    wide["change"] = wide["warehouse"] - wide["raw"]
    return wide


_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG.sub("_", text.lower()).strip("_")
