"""Load and validate config.toml."""
from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"

INDICATORS = [
    "single_bidder",
    "short_period",
    "non_competitive",
    "supplier_concentration",
    "award_close_to_estimate",
    "bid_level_screens",
]


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    raw: dict
    path: Path
    sha256: str
    root: Path

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def sources(self) -> list[dict]:
        return self.raw["sources"]

    def source(self, source_id: str) -> dict:
        for s in self.sources:
            if s["id"] == source_id:
                return s
        raise KeyError(source_id)

    def resolve(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def weights(self) -> dict[str, float]:
        w = {k: float(self.raw["indicators"][k]["weight"]) for k in INDICATORS}
        if self.raw["anomaly"]["enabled"]:
            w["anomaly"] = float(self.raw["anomaly"]["weight"])
        return w

    def method_class(self, method: str | None) -> str:
        return self.raw["methods"]["vocabulary"].get(method or "", "unknown")


def _validate(raw: dict) -> None:
    for section in ("paths", "run", "sources", "methods", "suppliers", "indicators", "anomaly", "prioritisation"):
        if section not in raw:
            raise ConfigError(f"missing [{section}] section")
    ids = [s["id"] for s in raw["sources"]]
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate source id")
    vocab = raw["methods"]["vocabulary"]
    for rule in raw["methods"]["detail_rules"]:
        if rule["method"] not in vocab:
            raise ConfigError(f"detail rule maps to unknown method {rule['method']!r}")
        re.compile(rule["pattern"])
    for code, method in raw["methods"]["code_map"].items():
        if method not in vocab:
            raise ConfigError(f"code_map[{code}] maps to unknown method {method!r}")
    for name in INDICATORS:
        if name not in raw["indicators"]:
            raise ConfigError(f"missing [indicators.{name}]")
        w = raw["indicators"][name]["weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ConfigError(f"weight for {name} must be a non-negative number")
    if raw["anomaly"]["weight"] < 0:
        raise ConfigError("anomaly weight must be non-negative")
    if sum(raw["indicators"][n]["weight"] for n in INDICATORS) <= 0:
        raise ConfigError("at least one indicator weight must be positive")


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = path.read_bytes()
    raw = tomllib.loads(data.decode("utf-8"))
    _validate(raw)
    return Config(raw=raw, path=path, sha256=hashlib.sha256(data).hexdigest(), root=root or path.parent)
