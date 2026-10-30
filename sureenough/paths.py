import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("SUREENOUGH_DATA", "D:/A5/metapr-data"))
RAW = DATA_ROOT / "raw"
DERIVED = DATA_ROOT / "derived"
OUTPUTS = REPO / "outputs"
HF_HOME = DATA_ROOT / "hf-cache"


def subject_ids() -> list[int]:
    return list(range(1, 11))


def use_project_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
