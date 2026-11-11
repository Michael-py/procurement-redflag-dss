"""Documented synthetic data.

Two uses, both clearly labelled as synthetic and never mixed into the real
warehouse:

1. Demonstrating indicators that the live data cannot exercise (an
   independent tender estimate for 'award close to estimate').
2. A mechanics check of the ranking: contracts are seeded with known
   combinations of flags and the watchlist is expected to surface them.

The generator is rule-based and seeded, so the same call always produces the
same releases. Every synthetic ocid starts with 'ocds-synth0-' and every name
with 'SYNTHETIC'.
"""
from __future__ import annotations

import copy
import json
import random
from datetime import date, timedelta
from pathlib import Path

from .config import Config

SOURCE_ID = "synthetic_demo"
SOURCE = {
    "id": SOURCE_ID,
    "label": "Synthetic demonstration data",
    "level": "synthetic",
    "file": "synthetic_demo.jsonl",
    "registry_publication": 0,
    "method_code_trusted": False,
    "blank_id_patterns": ["^SYN-$"],
}

N_BUYERS = 20
N_SUPPLIERS = 400
# Background rates loosely follow the federal data so the check is not trivial:
# many ordinary contracts carry one flag, and seeded cases must still outrank them.
BACKGROUND_METHODS = [
    ("National Competitive Bidding", 0.60),
    ("Selective Tendering", 0.20),
    ("Request for quotation", 0.08),
    ("Direct Procurement", 0.10),
    ("Emergency", 0.02),
]
BACKGROUND_SINGLE_BIDDER_RATE = 0.35
BACKGROUND_SHORT_PERIOD_RATE = 0.30


def _release(i: int, buyer: int, supplier: int, method: str, tenderers: int, days: int, amount: float,
             estimate: float, start: date) -> dict:
    ocid = f"ocds-synth0-{i:06d}"
    end = start + timedelta(days=days)
    award_date = end + timedelta(days=21)
    return {
        "ocid": ocid,
        "id": f"{ocid}-1",
        "date": award_date.isoformat() + "T00:00:00Z",
        "tag": ["compiled"],
        "initiationType": "tender",
        "buyer": {"id": f"SYN-B{buyer:03d}", "name": f"SYNTHETIC BUYER {buyer:03d}"},
        "tender": {
            "id": f"T{i:06d}",
            "title": f"SYNTHETIC contract {i:06d}",
            "status": "complete",
            "procurementMethod": "open",
            "procurementMethodDetails": method,
            "numberOfTenderers": tenderers,
            "tenderPeriod": {"startDate": start.isoformat() + "T00:00:00Z", "endDate": end.isoformat() + "T00:00:00Z", "durationInDays": days},
            "value": {"amount": round(estimate, 2), "currency": "NGN"},
        },
        "awards": [
            {
                "id": f"A{i:06d}",
                "date": award_date.isoformat() + "T00:00:00Z",
                "status": "active",
                "value": {"amount": round(amount, 2), "currency": "NGN"},
                "suppliers": [{"id": f"SYN-S{supplier:04d}", "name": f"SYNTHETIC SUPPLIER {supplier:04d} LTD"}],
            }
        ],
    }


def generate(n_background: int = 1900, n_seeded: int = 100, seed: int = 20261012) -> tuple[list[dict], set[str]]:
    """Return (releases, seeded_ocids).

    Background contracts get flags independently at realistic rates. Seeded
    contracts combine a single bidder under National Competitive Bidding, a
    submission period under 14 days and an award within 1% of an independent
    estimate: three rules at once, which chance alone rarely produces.
    """
    rng = random.Random(seed)
    methods, weights = zip(*BACKGROUND_METHODS)
    base = date(2023, 1, 2)
    releases, seeded = [], set()
    kinds = ["seeded"] * n_seeded + ["background"] * n_background
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds, start=1):
        buyer = rng.randrange(1, N_BUYERS + 1)
        supplier = rng.randrange(1, N_SUPPLIERS + 1)
        start = base + timedelta(days=rng.randrange(0, 700))
        amount = 10 ** rng.uniform(6.0, 9.5)
        if kind == "seeded":
            method, tenderers, days = "National Competitive Bidding", 1, rng.randrange(2, 14)
            estimate = amount * (1 + rng.uniform(0.001, 0.009))
            seeded.add(f"ocds-synth0-{i:06d}")
        else:
            method = rng.choices(methods, weights)[0]
            tenderers = 1 if rng.random() < BACKGROUND_SINGLE_BIDDER_RATE else rng.randrange(2, 9)
            days = rng.randrange(5, 42) if rng.random() < BACKGROUND_SHORT_PERIOD_RATE else rng.randrange(42, 75)
            estimate = amount * (1 + rng.uniform(0.03, 0.25))
        releases.append(_release(i, buyer, supplier, method, tenderers, days, amount, estimate, start))
    return releases, seeded


def synthetic_config(cfg: Config) -> Config:
    """A copy of the configuration whose only source is the synthetic one."""
    raw = copy.deepcopy(cfg.raw)
    raw["sources"] = [dict(SOURCE)]
    return Config(raw=raw, path=cfg.path, sha256=cfg.sha256, root=cfg.root)


def write_synthetic_file(cfg: Config) -> Path:
    releases, seeded = generate()
    out = cfg.root / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    path = out / SOURCE["file"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in releases:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    (out / "seeded_ocids.txt").write_text("\n".join(sorted(seeded)) + "\n", encoding="utf-8")
    return path
