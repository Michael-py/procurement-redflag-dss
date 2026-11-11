"""Consolidated SQLite warehouse.

The schema follows the entity model in Figure 3.4. Every table is keyed by
the OCDS contracting-process identifier (`ocid`) or references it. The BID
table exists but stays empty: no Nigerian source publishes bid-level data.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE run_manifest (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE source_file (
    source      TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    level       TEXT NOT NULL,
    file        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    bytes       INTEGER NOT NULL,
    lines       INTEGER NOT NULL,
    accepted    INTEGER NOT NULL,
    rejected    INTEGER NOT NULL
);

CREATE TABLE buyer (
    buyer_key    TEXT PRIMARY KEY,
    source       TEXT NOT NULL REFERENCES source_file(source),
    buyer_id_raw TEXT,
    name         TEXT
);

CREATE TABLE supplier (
    supplier_key    TEXT PRIMARY KEY,
    source          TEXT NOT NULL REFERENCES source_file(source),
    name            TEXT,
    n_name_variants INTEGER,
    n_awards        INTEGER
);

CREATE TABLE contracting_process (
    ocid           TEXT PRIMARY KEY,
    source         TEXT NOT NULL REFERENCES source_file(source),
    release_date   TEXT,
    buyer_key      TEXT REFERENCES buyer(buyer_key),
    has_planning   INTEGER NOT NULL,
    has_tender     INTEGER NOT NULL,
    n_awards       INTEGER NOT NULL,
    n_contracts    INTEGER NOT NULL,
    n_transactions INTEGER NOT NULL
);

CREATE TABLE tender (
    ocid               TEXT PRIMARY KEY REFERENCES contracting_process(ocid),
    tender_id          TEXT,
    title              TEXT,
    status             TEXT,
    method_code_raw    TEXT,
    method_details_raw TEXT,
    method             TEXT,
    method_class       TEXT NOT NULL,
    method_origin      TEXT NOT NULL,
    num_tenderers      INTEGER,
    period_start       TEXT,
    period_end         TEXT,
    period_days        INTEGER,           -- recomputed from the dates
    period_days_published INTEGER,        -- durationInDays as published
    estimate_amount    REAL,
    category           TEXT
);

CREATE TABLE award (
    award_key           TEXT PRIMARY KEY,
    ocid                TEXT NOT NULL REFERENCES contracting_process(ocid),
    source              TEXT NOT NULL REFERENCES source_file(source),
    award_id            TEXT,
    title               TEXT,
    award_date          TEXT,
    award_date_raw      TEXT,
    amount              REAL,
    currency            TEXT,
    status              TEXT,
    supplier_id_raw     TEXT,
    supplier_name_raw   TEXT,
    supplier_id_blank   INTEGER NOT NULL,
    supplier_name_norm  TEXT,
    estimate_copied     INTEGER NOT NULL,
    supplier_key        TEXT REFERENCES supplier(supplier_key),
    supplier_resolution TEXT NOT NULL
);

-- Present for extensibility; no current source populates it (Section 3.5.1).
CREATE TABLE bid (
    bid_id       TEXT PRIMARY KEY,
    ocid         TEXT NOT NULL REFERENCES tender(ocid),
    supplier_key TEXT REFERENCES supplier(supplier_key),
    bid_value    REAL,
    status       TEXT
);

CREATE TABLE validation_log (
    source   TEXT NOT NULL,
    ocid     TEXT,
    line_no  INTEGER,
    severity TEXT NOT NULL,
    code     TEXT NOT NULL,
    detail   TEXT
);

CREATE TABLE dq_metric (
    source      TEXT NOT NULL,
    stage       TEXT NOT NULL,      -- 'raw' or 'warehouse'
    dimension   TEXT NOT NULL,      -- completeness / validity / consistency / fitness
    metric      TEXT NOT NULL,
    numerator   INTEGER NOT NULL,
    denominator INTEGER NOT NULL,
    value       REAL,
    note        TEXT,
    PRIMARY KEY (source, stage, dimension, metric)
);

CREATE TABLE indicator_result (
    ocid      TEXT NOT NULL REFERENCES contracting_process(ocid),
    indicator TEXT NOT NULL,
    status    TEXT NOT NULL,        -- flagged / clear / not_applicable / non_computable
    value     REAL,                 -- flag strength in [0, 1]; NULL unless flagged or clear
    reason    TEXT,
    PRIMARY KEY (ocid, indicator)
);

CREATE TABLE contract_score (
    ocid            TEXT PRIMARY KEY REFERENCES contracting_process(ocid),
    rank            INTEGER NOT NULL,
    risk_score      REAL NOT NULL,
    n_flags         INTEGER NOT NULL,
    coverage        REAL NOT NULL,
    anomaly_score   REAL,
    explanation     TEXT,
    not_assessed    TEXT
);

CREATE INDEX ix_process_source ON contracting_process(source);
CREATE INDEX ix_process_buyer ON contracting_process(buyer_key);
CREATE INDEX ix_award_ocid ON award(ocid);
CREATE INDEX ix_award_supplier ON award(supplier_key);
CREATE INDEX ix_log_code ON validation_log(source, code);
CREATE INDEX ix_indicator ON indicator_result(indicator, status);
CREATE INDEX ix_score_rank ON contract_score(rank);
"""

# Load order respects foreign keys.
TABLE_ORDER = [
    "source_file", "buyer", "supplier", "contracting_process", "tender", "award",
    "validation_log", "dq_metric", "indicator_result", "contract_score",
]


def write_warehouse(path: Path, tables: dict[str, pd.DataFrame], manifest: dict[str, str],
                    marts: dict[str, pd.DataFrame] | None = None) -> None:
    """Build the warehouse in a temporary file and swap it in atomically, so a
    failed run never leaves a half-written database behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with closing(sqlite3.connect(tmp)) as con:
        con.executescript(DDL)
        with con:
            con.executemany("INSERT INTO run_manifest(key, value) VALUES (?, ?)", sorted(manifest.items()))
            for name in TABLE_ORDER:
                df = tables.get(name)
                if df is None or df.empty:
                    continue
                df = df.astype(object).where(df.notna(), None)
                for col in df.columns:
                    df[col] = df[col].map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
                placeholders = ", ".join("?" for _ in df.columns)
                con.executemany(
                    f"INSERT INTO {name} ({', '.join(df.columns)}) VALUES ({placeholders})",
                    df.itertuples(index=False, name=None),
                )
        for name, df in (marts or {}).items():  # denormalised read-only views for the dashboard
            mart = df.copy()
            for col in mart.columns:
                if mart[col].dtype == object:
                    mart[col] = mart[col].map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
            mart.to_sql(name, con, index=False)
        con.commit()
        problems = con.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"foreign-key violations in warehouse: {problems[:5]}")
    os.replace(tmp, path)


def connect(path: Path, read_only: bool = True) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    return sqlite3.connect(path)


def read_table(path: Path, table_or_sql: str, params: tuple = ()) -> pd.DataFrame:
    sql = table_or_sql if " " in table_or_sql.strip() else f"SELECT * FROM {table_or_sql}"
    with closing(connect(path)) as con:
        return pd.read_sql_query(sql, con, params=params)


def contracts_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per contracting process that reached the tender or award stage.

    This is the unit that indicators score and the watchlist ranks. A process
    with several awards carries their total value and its largest award's
    supplier.
    """
    proc = tables["contracting_process"]
    tender = tables["tender"].drop(columns=["tender_id"], errors="ignore")
    award = tables["award"].copy()
    live = award[~award["status"].isin(["cancelled", "unsuccessful"])]
    live = live.sort_values(["ocid", "amount", "award_key"], ascending=[True, False, True], na_position="last")
    agg = live.groupby("ocid").agg(
        award_amount=("amount", lambda s: s.sum(min_count=1)),
        award_date=("award_date", "first"),
        supplier_key=("supplier_key", "first"),
        supplier_name=("supplier_name_raw", "first"),
        supplier_resolution=("supplier_resolution", "first"),
        estimate_copied=("estimate_copied", "max"),
        n_live_awards=("award_key", "count"),
        award_title=("title", "first"),
    )
    df = proc[(proc["has_tender"] == 1) | (proc["n_awards"] > 0)].merge(tender, on="ocid", how="left").merge(agg, on="ocid", how="left")
    df = df.merge(tables["buyer"][["buyer_key", "name"]].rename(columns={"name": "buyer_name"}), on="buyer_key", how="left")
    df["title"] = df["title"].fillna(df["award_title"])
    df["method_class"] = df["method_class"].fillna("unknown")
    df["n_live_awards"] = df["n_live_awards"].fillna(0).astype(int)
    return df.drop(columns=["award_title"]).sort_values("ocid").reset_index(drop=True)
