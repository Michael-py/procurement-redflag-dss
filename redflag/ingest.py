"""Ingestion and structural validation of OCDS JSON Lines files.

Each line of a registry bulk file is one compiled release. A line is rejected
(and written to the validation log) if it cannot be used as a contracting
process at all; field-level defects are handled later, during harmonisation,
so that a single bad field does not discard an otherwise usable record.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ValidationLog:
    """Collects every rejected record and field-level repair."""

    rows: list[dict] = field(default_factory=list)

    def add(self, source: str, ocid: str | None, severity: str, code: str, detail: str = "", line_no: int | None = None):
        self.rows.append(
            {"source": source, "ocid": ocid, "line_no": line_no, "severity": severity, "code": code, "detail": str(detail)[:300]}
        )

    def count(self, code: str | None = None, source: str | None = None) -> int:
        return sum(1 for r in self.rows if (code is None or r["code"] == code) and (source is None or r["source"] == source))


@dataclass
class IngestStats:
    source: str
    file: str
    sha256: str
    bytes: int
    lines: int = 0
    accepted: int = 0
    rejected: int = 0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def validate_release(rec) -> str | None:
    """Return a rejection code, or None if the record is structurally usable."""
    if not isinstance(rec, dict):
        return "not_an_object"
    ocid = rec.get("ocid")
    if not isinstance(ocid, str) or not ocid.strip():
        return "missing_ocid"
    if not ocid.startswith("ocds-"):
        return "malformed_ocid"
    for key in ("tender", "planning", "buyer"):
        if rec.get(key) is not None and not isinstance(rec[key], dict):
            return f"{key}_not_an_object"
    for key in ("awards", "contracts", "parties"):
        if rec.get(key) is not None and not isinstance(rec[key], list):
            return f"{key}_not_a_list"
    return None


def read_source(path: Path, source_id: str, log: ValidationLog) -> tuple[Iterator[dict], IngestStats]:
    """Yield structurally valid releases from one bulk file.

    Statistics are filled in as the iterator is consumed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"source file for {source_id} not found: {path}")
    stats = IngestStats(source=source_id, file=path.name, sha256=sha256_of(path), bytes=path.stat().st_size)

    def gen():
        seen: set[str] = set()
        with _open_text(path) as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                stats.lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.rejected += 1
                    log.add(source_id, None, "error", "invalid_json", exc.msg, line_no)
                    continue
                code = validate_release(rec)
                if code:
                    stats.rejected += 1
                    log.add(source_id, rec.get("ocid") if isinstance(rec, dict) else None, "error", code, "", line_no)
                    continue
                if rec["ocid"] in seen:
                    stats.rejected += 1
                    log.add(source_id, rec["ocid"], "error", "duplicate_ocid", "", line_no)
                    continue
                seen.add(rec["ocid"])
                stats.accepted += 1
                yield rec

    return gen(), stats
