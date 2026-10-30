import hashlib
import json
from pathlib import Path

from sureenough.paths import REPO

PRIMARY = REPO / "configs" / "primary.json"


def _digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def freeze(cfg: dict, path: Path = PRIMARY) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    digest = _digest(cfg)
    path.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def load_frozen(path: Path = PRIMARY) -> dict:
    cfg = json.loads(path.read_text())
    if path.with_suffix(".sha256").read_text().strip() != _digest(cfg):
        raise ValueError(f"{path.name} changed after it was frozen")
    return cfg
