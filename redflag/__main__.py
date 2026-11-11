"""Command-line entry point.

    python -m redflag run        ingest, harmonise, score and export
    python -m redflag evaluate   run the four-part evaluation (Chapter 5)
    python -m redflag synth      write the labelled synthetic demonstration file
    python -m redflag pg-load    mirror the warehouse into PostgreSQL (see docker-compose.yml)
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redflag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "evaluate", "synth", "pg-load"])
    parser.add_argument("--config", default=None, help="path to config.toml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "run":
        from .pipeline import run

        run(args.config)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(args.config)
    elif args.command == "pg-load":
        from .config import load_config
        from .pgload import load

        for table, rows in load(load_config(args.config)).items():
            print(f"{table:22s} {rows:>8,d}")
    else:
        from .config import load_config
        from .synthetic import write_synthetic_file

        print(write_synthetic_file(load_config(args.config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
