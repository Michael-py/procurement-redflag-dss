"""Standardisation and semantic harmonisation.

Turns validated OCDS releases into flat, typed rows for the warehouse and
repairs or flags the semantic defects found in Nigerian publications:

* procurement methods are recovered from free text and mapped to a controlled
  vocabulary (`map_method`);
* dates are parsed and implausible ones are nulled, keeping the raw value;
* statuses are normalised to the OCDS code list;
* supplier identities are resolved when identifiers are blank
  (`resolve_suppliers`).

Every repair is written to the validation log so that nothing is changed
silently (NFR6).
"""
from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

from .config import Config
from .ingest import ValidationLog

_STATUS_MAP = {
    "completed": "complete",
    "ongoing": "active",
}
_TENDER_STATUSES = {"planning", "planned", "active", "cancelled", "unsuccessful", "complete", "withdrawn"}
_AWARD_STATUSES = {"pending", "active", "cancelled", "unsuccessful", "complete"}


# ------------------------------------------------------------------ scalars
def parse_date(value) -> tuple[date | None, str]:
    """Parse an OCDS date-time. Returns (date, status) with status one of
    'ok', 'missing', 'invalid'."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    if not isinstance(value, str) or len(value) < 10:
        return None, "invalid"
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date(), "ok"
    except ValueError:
        try:
            return date.fromisoformat(value[:10]), "ok"
        except ValueError:
            return None, "invalid"


def clean_amount(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def clean_count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def normalise_status(value, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    v = _STATUS_MAP.get(v, v)
    return v if v in allowed else None


# ------------------------------------------------------------------ methods
def map_method(details, code, source_cfg: dict, cfg: Config) -> tuple[str | None, str]:
    """Map a tender's published method to the controlled vocabulary.

    Returns (method, origin) where origin is 'details', 'code' or 'unmapped'.
    The free-text details are tried first because NOCOPO publishes the real
    Nigerian method only there; the OCDS code is a fallback for sources where
    it is trusted.
    """
    if isinstance(details, str) and details.strip():
        text = details.strip().lower()
        for rule in cfg["methods"]["detail_rules"]:
            if re.search(rule["pattern"], text):
                return rule["method"], "details"
    if source_cfg.get("method_code_trusted") and isinstance(code, str):
        method = cfg["methods"]["code_map"].get(code.strip().lower())
        if method:
            return method, "code"
    return None, "unmapped"


# -------------------------------------------------------------------- names
_PUNCT = re.compile(r"[^A-Z0-9 ]+")


def normalise_name(name, legal_tokens: set[str]) -> str:
    """Upper-case, strip punctuation and legal-form tokens, collapse spaces."""
    if not isinstance(name, str):
        return ""
    text = _PUNCT.sub(" ", name.upper().replace("&", " AND "))
    tokens = [t for t in text.split() if t not in legal_tokens]
    return " ".join(tokens)


def is_blank_id(identifier, source_cfg: dict) -> bool:
    if not isinstance(identifier, str) or not identifier.strip():
        return True
    return any(re.search(p, identifier.strip()) for p in source_cfg.get("blank_id_patterns", []))


# ------------------------------------------------------------------ flatten
class Harmoniser:
    """Flattens releases from one or more sources into row lists."""

    def __init__(self, cfg: Config, log: ValidationLog):
        self.cfg = cfg
        self.log = log
        self.min_date = date.fromisoformat(cfg["run"]["min_plausible_date"])
        self.max_date = date.fromisoformat(cfg["run"]["retrieval_date"])
        self.legal = set(cfg["suppliers"]["legal_form_tokens"])
        self.processes: list[dict] = []
        self.tenders: list[dict] = []
        self.awards: list[dict] = []
        self.buyers: dict[str, dict] = {}

    # -- helpers
    def _date(self, value, source, ocid, field):
        d, status = parse_date(value)
        if status == "invalid":
            self.log.add(source, ocid, "warning", "invalid_date", f"{field}={value!r}")
            return None
        if d is not None and not (self.min_date <= d <= self.max_date):
            self.log.add(source, ocid, "warning", "implausible_date", f"{field}={value!r}")
            return None
        return d

    def _status(self, value, allowed, source, ocid, field):
        status = normalise_status(value, allowed)
        if value not in (None, "") and status is None:
            self.log.add(source, ocid, "warning", "invalid_status", f"{field}={value!r}")
        elif isinstance(value, str) and status is not None and status != value:
            self.log.add(source, ocid, "info", "status_normalised", f"{field}: {value!r} -> {status!r}")
        return status

    def _buyer(self, rec, source_cfg) -> tuple[str | None, str | None]:
        buyer = rec.get("buyer") or (rec.get("tender") or {}).get("procuringEntity") or {}
        name = buyer.get("name").strip() if isinstance(buyer.get("name"), str) else None
        bid = buyer.get("id")
        source = source_cfg["id"]
        if not is_blank_id(bid, source_cfg):
            key = f"{source}:{bid.strip()}"
        elif name:
            key = f"{source}:NAME:{normalise_name(name, set())}"
        else:
            return None, None
        entry = self.buyers.setdefault(key, {"buyer_key": key, "source": source, "buyer_id_raw": bid, "name": name})
        if entry["name"] is None and name:
            entry["name"] = name
        return key, name

    # -- main
    def add(self, rec: dict, source_cfg: dict) -> None:
        source = source_cfg["id"]
        ocid = rec["ocid"]
        tender = rec.get("tender") or {}
        awards = [a for a in (rec.get("awards") or []) if isinstance(a, dict)]
        contracts = [c for c in (rec.get("contracts") or []) if isinstance(c, dict)]
        buyer_key, _ = self._buyer(rec, source_cfg)
        if buyer_key is None:
            self.log.add(source, ocid, "warning", "missing_buyer")

        n_tx = sum(len(((c.get("implementation") or {}).get("transactions")) or []) for c in contracts)
        self.processes.append(
            {
                "ocid": ocid,
                "source": source,
                "release_date": self._date(rec.get("date"), source, ocid, "date"),
                "buyer_key": buyer_key,
                "has_planning": int(bool(rec.get("planning"))),
                "has_tender": int(bool(tender)),
                "n_awards": len(awards),
                "n_contracts": len(contracts),
                "n_transactions": n_tx,
            }
        )

        estimate = None
        if tender:
            estimate = self._tender(tender, source_cfg, ocid)
        for idx, award in enumerate(awards):
            self._award(award, idx, source_cfg, ocid, estimate)

    def _tender(self, tender: dict, source_cfg: dict, ocid: str) -> float | None:
        source = source_cfg["id"]
        details, code = tender.get("procurementMethodDetails"), tender.get("procurementMethod")
        method, origin = map_method(details, code, source_cfg, self.cfg)
        if method is None:
            self.log.add(source, ocid, "warning", "method_unmapped", f"details={details!r} code={code!r}")

        raw_n = tender.get("numberOfTenderers")
        n = clean_count(raw_n)
        if raw_n is not None and n is None:
            self.log.add(source, ocid, "warning", "invalid_number_of_tenderers", repr(raw_n))

        period = tender.get("tenderPeriod") or {}
        start = self._date(period.get("startDate"), source, ocid, "tender.tenderPeriod.startDate")
        end = self._date(period.get("endDate"), source, ocid, "tender.tenderPeriod.endDate")
        days = None
        published = period.get("durationInDays")
        published = int(published) if isinstance(published, (int, float)) and not isinstance(published, bool) else None
        if start and end:
            if end < start:
                self.log.add(source, ocid, "warning", "period_end_before_start", f"{start}..{end}")
            else:
                days = (end - start).days
                if published is not None and published != days:
                    self.log.add(source, ocid, "info", "period_duration_recomputed", f"published={published} computed={days}")

        value = tender.get("value") or {}
        estimate = clean_amount(value.get("amount"))
        self.tenders.append(
            {
                "ocid": ocid,
                "tender_id": str(tender.get("id")) if tender.get("id") is not None else None,
                "title": (tender.get("title") or tender.get("description") or None),
                "status": self._status(tender.get("status"), _TENDER_STATUSES, source, ocid, "tender.status"),
                "method_code_raw": code if isinstance(code, str) else None,
                "method_details_raw": details if isinstance(details, str) else None,
                "method": method,
                "method_class": self.cfg.method_class(method),
                "method_origin": origin,
                "num_tenderers": n,
                "period_start": start,
                "period_end": end,
                "period_days": days,
                "period_days_published": published,
                "estimate_amount": estimate,
                "category": tender.get("mainProcurementCategory") or tender.get("procurementCategory"),
            }
        )
        return estimate

    def _award(self, award: dict, idx: int, source_cfg: dict, ocid: str, estimate: float | None) -> None:
        source = source_cfg["id"]
        value = award.get("value") or {}
        amount = clean_amount(value.get("amount"))
        if value.get("amount") is not None and amount is None:
            self.log.add(source, ocid, "warning", "invalid_amount", f"award.value.amount={value.get('amount')!r}")
        currency = value.get("currency")
        if amount is not None and not currency:
            currency = self.cfg["run"]["default_currency"]
            self.log.add(source, ocid, "info", "currency_assumed", currency)

        suppliers = [s for s in (award.get("suppliers") or []) if isinstance(s, dict)]
        supplier = suppliers[0] if suppliers else {}
        if len(suppliers) > 1:
            self.log.add(source, ocid, "info", "multiple_suppliers", f"{len(suppliers)} suppliers; first used")
        sid = supplier.get("id") if isinstance(supplier.get("id"), str) else None
        sname = supplier.get("name").strip() if isinstance(supplier.get("name"), str) and supplier.get("name").strip() else None

        award_id = str(award.get("id")) if award.get("id") is not None else str(idx)
        self.awards.append(
            {
                "award_key": f"{ocid}#{award_id}#{idx}",
                "ocid": ocid,
                "source": source,
                "award_id": award_id,
                "title": award.get("title") or award.get("description") or None,
                "award_date": self._date(award.get("date"), source, ocid, "awards.date"),
                "award_date_raw": award.get("date") if isinstance(award.get("date"), str) else None,
                "amount": amount,
                "currency": currency if amount is not None else None,
                "status": self._status(award.get("status"), _AWARD_STATUSES, source, ocid, "awards.status"),
                "supplier_id_raw": sid,
                "supplier_name_raw": sname,
                "supplier_id_blank": int(is_blank_id(sid, source_cfg)),
                "supplier_name_norm": normalise_name(sname, self.legal),
                "estimate_copied": int(amount is not None and estimate is not None and amount == estimate),
            }
        )

    # -- results
    def frames(self) -> dict[str, pd.DataFrame]:
        process_cols = ["ocid", "source", "release_date", "buyer_key", "has_planning", "has_tender", "n_awards", "n_contracts", "n_transactions"]
        tender_cols = ["ocid", "tender_id", "title", "status", "method_code_raw", "method_details_raw", "method", "method_class",
                       "method_origin", "num_tenderers", "period_start", "period_end", "period_days", "period_days_published", "estimate_amount", "category"]
        award_cols = ["award_key", "ocid", "source", "award_id", "title", "award_date", "award_date_raw", "amount", "currency", "status",
                      "supplier_id_raw", "supplier_name_raw", "supplier_id_blank", "supplier_name_norm", "estimate_copied"]
        awards = pd.DataFrame(self.awards, columns=award_cols)
        awards, suppliers = resolve_suppliers(awards, self.cfg, self.log)
        tenders = pd.DataFrame(self.tenders, columns=tender_cols)
        tenders["num_tenderers"] = tenders["num_tenderers"].astype("Int64")
        tenders["period_days"] = tenders["period_days"].astype("Int64")
        tenders["period_days_published"] = tenders["period_days_published"].astype("Int64")
        return {
            "contracting_process": pd.DataFrame(self.processes, columns=process_cols),
            "tender": tenders,
            "award": awards,
            "buyer": pd.DataFrame(list(self.buyers.values()), columns=["buyer_key", "source", "buyer_id_raw", "name"]),
            "supplier": suppliers,
        }


# ------------------------------------------------------- supplier resolution
def resolve_suppliers(awards: pd.DataFrame, cfg: Config, log: ValidationLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every award a `supplier_key`, resolving blank identifiers.

    1. An award with an informative identifier keeps it ('identifier').
    2. An award with a blank identifier whose normalised name matches exactly
       one identified supplier in the same source inherits that supplier
       ('name_match').
    3. Any other award with a usable name is keyed by that name ('name_only').
    4. An award with neither is left 'unresolved' and is excluded from the
       concentration indicator rather than lumped into one fake supplier.
    """
    awards = awards.copy()
    min_len = cfg["suppliers"]["min_name_length"]
    has_id = awards["supplier_id_blank"] == 0
    has_name = awards["supplier_name_norm"].str.len() >= min_len

    awards["supplier_key"] = None
    awards["supplier_resolution"] = "unresolved"
    awards.loc[has_id, "supplier_key"] = awards["source"] + ":" + awards["supplier_id_raw"].fillna("").str.strip()
    awards.loc[has_id, "supplier_resolution"] = "identifier"

    # name -> identifier lookup, kept only where a name maps to a single identifier
    known = awards[has_id & has_name].groupby(["source", "supplier_name_norm"])["supplier_key"].agg(lambda s: s.unique().tolist())
    unique_lookup = {k: v[0] for k, v in known.items() if len(v) == 1}

    blank_named = ~has_id & has_name
    for i in awards.index[blank_named]:
        key = (awards.at[i, "source"], awards.at[i, "supplier_name_norm"])
        if key in unique_lookup:
            awards.at[i, "supplier_key"] = unique_lookup[key]
            awards.at[i, "supplier_resolution"] = "name_match"
        else:
            awards.at[i, "supplier_key"] = f"{key[0]}:NAME:{key[1]}"
            awards.at[i, "supplier_resolution"] = "name_only"

    if log is not None:
        for row in awards[awards["supplier_resolution"] == "unresolved"].itertuples():
            log.add(row.source, row.ocid, "warning", "supplier_unresolved", f"id={row.supplier_id_raw!r} name={row.supplier_name_raw!r}")

    resolved = awards[awards["supplier_key"].notna()]
    suppliers = (
        resolved.groupby("supplier_key")
        .agg(
            source=("source", "first"),
            name=("supplier_name_raw", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
            n_name_variants=("supplier_name_norm", lambda s: s[s != ""].nunique()),
            n_awards=("award_key", "count"),
        )
        .reset_index()
    )
    return awards, suppliers
