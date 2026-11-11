# Procurement Red-Flag DSS

A self-hostable pipeline and decision-support system for Nigerian public procurement data. It consolidates the
federal (NOCOPO / BPP) and four state OCDS publications into one SQLite warehouse, measures how fit each field is
for analysis, computes the red-flag indicators the data can support, and ranks contracts for review with a
plain-language reason for each.

**Flags are risk signals that prioritise contracts for human review. They are not findings or allegations of
wrongdoing.**

## Run it

Requires Python 3.11 or later. From this folder (Windows shown; on Linux or macOS use `.venv/bin/python`):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m redflag run          # build data/warehouse.sqlite and outputs/
.venv\Scripts\python -m streamlit run app.py # open the dashboard at http://localhost:8501
```

A full run takes about 25 seconds on a laptop.

Other commands:

```
.venv\Scripts\python -m pytest tests         # 113 unit and integration tests
.venv\Scripts\python -m redflag evaluate     # regenerate outputs/evaluation/ (the report's Chapter 5 results)
.venv\Scripts\python -m redflag synth        # write the labelled synthetic demonstration file
```

## PostgreSQL in Docker (optional)

SQLite is the default and needs no setup. For a server database, a PostgreSQL 16 container is provided.
Requires Docker Desktop.

```
copy .env.example .env                        # then set two passwords in .env
docker compose up -d                          # start PostgreSQL on 127.0.0.1:5432
.venv\Scripts\python -m redflag run           # build the SQLite warehouse (as usual)
.venv\Scripts\python -m redflag pg-load       # mirror it into PostgreSQL, schema "redflag"
```

To run the dashboard on PostgreSQL instead of the SQLite file (PowerShell):

```
$env:REDFLAG_DB = "postgres"; .venv\Scripts\python -m streamlit run app.py
```

- `pg-load` replaces the whole schema in one transaction and checks row counts before committing.
- The dashboard connects as `redflag_reader`, which can only read.
- The port is published on 127.0.0.1 only. `docker compose down` stops the database and keeps the data;
  `docker compose down -v` deletes it.
- The reader login is created when the data volume is first created. If you change `READER_PASSWORD`
  later, run `docker compose down -v` and start again.

## Data

`data/raw/` holds the five bulk files downloaded from the OCP Data Registry
(`https://data.open-contracting.org/en/publication/<id>/download?name=full.jsonl.gz`, publications 64, 127, 86,
118 and 125). To refresh, download them again and set `retrieval_date` in `config.toml`. Each run records the
SHA-256 of every input file and of the configuration in the warehouse.

`data/reference/` holds the issue list extracted from the Auditor-General's 2024 annual report and its matching to
NOCOPO buyers, used only by the external plausibility check. `tools/audit_extraction/` contains the scripts that
produced it (Windows OCR of the scanned PDFs, then parsing).

## Configuration

Everything adjustable is in `config.toml`: sources, the procurement-method vocabulary and mapping rules,
indicator thresholds and weights, and the anomaly layer. Adding a source that publishes OCDS means adding one
`[[sources]]` block.

## Layout

```
config.toml        settings
app.py             dashboard
redflag/           the pipeline package (ingest, harmonise, quality, warehouse, indicators, anomaly,
                   prioritise, synthetic, evaluate, pipeline)
tests/             tests
tools/             one-off extraction scripts
data/              inputs, reference data and the warehouse
outputs/           watchlist, quality tables and evaluation results
```

## Known limits

- The watchlist has not been validated against real outcomes or tested with users.
- "Award close to estimate" is computable only for Plateau State; bid-level screens for no source.
- Submission-period thresholds come from the federal Public Procurement Act and are applied to state data as a
  common benchmark.
- The dashboard has no login. It is bound to `localhost`; do not expose it to a network.
