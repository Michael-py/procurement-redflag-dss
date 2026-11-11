"""Mirror the SQLite warehouse into PostgreSQL.

    python -m redflag pg-load

The pipeline always builds the SQLite file first; this step copies it, table
for table, into a PostgreSQL schema for deployments that need a server
(several users, remote access, larger volumes). The whole load runs in one
transaction: readers see either the previous warehouse or the new one, never
a half-loaded state, which is the server-side equivalent of the atomic file
swap used for SQLite.

Connection settings are read from the environment or from a .env file next
to config.toml: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST,
POSTGRES_PORT, READER_PASSWORD.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import Config
from .warehouse import DDL

log = logging.getLogger("redflag.pg")

SCHEMA = "redflag"
READER = "redflag_reader"
_TYPE_MAP = {"INTEGER": "BIGINT", "REAL": "DOUBLE PRECISION", "TEXT": "TEXT"}


def read_env(root: Path) -> dict[str, str]:
    """Environment variables, with a .env file as fallback."""
    values: dict[str, str] = {}
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith(("POSTGRES_", "READER_"))})
    return values


def conninfo(root: Path, reader: bool = False) -> str:
    env = read_env(root)
    password = env.get("READER_PASSWORD" if reader else "POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("PostgreSQL password not set: copy .env.example to .env and fill it in")
    user = READER if reader else env.get("POSTGRES_USER", "redflag")
    return (f"host={env.get('POSTGRES_HOST', '127.0.0.1')} port={env.get('POSTGRES_PORT', '5432')} "
            f"dbname={env.get('POSTGRES_DB', 'redflag')} user={user} password={password} "
            f"options=-csearch_path={SCHEMA}")


def postgres_ddl() -> str:
    """The warehouse schema in PostgreSQL's dialect. SQLite's REAL is an
    8-byte float; PostgreSQL's REAL is 4 bytes and would round naira amounts."""
    ddl = "\n".join(line for line in DDL.splitlines() if not line.strip().upper().startswith("PRAGMA"))
    ddl = re.sub(r"\bREAL\b", "DOUBLE PRECISION", ddl)
    return re.sub(r"\bINTEGER\b", "BIGINT", ddl)


def _mart_ddl(con: sqlite3.Connection, table: str) -> str:
    cols = con.execute(f"PRAGMA table_info({table})").fetchall()
    defs = ", ".join(f'"{c[1]}" {_TYPE_MAP.get((c[2] or "TEXT").upper(), "TEXT")}' for c in cols)
    return f"CREATE TABLE {table} ({defs})"


def load(cfg: Config) -> dict[str, int]:
    """Copy every warehouse table into PostgreSQL. Returns rows per table."""
    import psycopg  # imported here so the SQLite-only workflow needs no driver

    sqlite_path = cfg.resolve("warehouse")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"{sqlite_path} not found: run `python -m redflag run` first")

    counts: dict[str, int] = {}
    with closing(sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)) as src, \
            psycopg.connect(conninfo(cfg.root)) as dst:
        declared = set(re.findall(r"CREATE TABLE (\w+)", DDL))
        # sqlite_master lists tables in creation order, which respects the foreign keys
        tables = [r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid")]
        with dst.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            cur.execute(f"CREATE SCHEMA {SCHEMA}")
            cur.execute(f"SET LOCAL search_path TO {SCHEMA}")
            cur.execute(postgres_ddl())
            for table in tables:
                if table not in declared:
                    cur.execute(_mart_ddl(src, table))
                cols = [c[1] for c in src.execute(f"PRAGMA table_info({table})")]
                col_list = ", ".join(f'"{c}"' for c in cols)
                n = 0
                with cur.copy(f"COPY {table} ({col_list}) FROM STDIN") as copy:
                    for row in src.execute(f"SELECT {col_list} FROM {table}"):
                        copy.write_row(row)
                        n += 1
                counts[table] = n
            cur.execute("CREATE INDEX ix_mart_source ON contract_mart(source)")
            cur.execute("CREATE UNIQUE INDEX ix_mart_ocid ON contract_mart(ocid)")
            # verify before committing: the mirror must match the source row for row
            for table, expected in counts.items():
                actual = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if actual != expected:
                    raise RuntimeError(f"{table}: {actual} rows in PostgreSQL, {expected} in SQLite")
            if cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (READER,)).fetchone():
                cur.execute(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {READER}")
                cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO {READER}")
            cur.execute("ANALYZE")
        dst.commit()
    log.info("loaded %d tables, %d rows into PostgreSQL schema '%s'", len(counts), sum(counts.values()), SCHEMA)
    return counts
