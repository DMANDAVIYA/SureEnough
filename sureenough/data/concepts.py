import csv
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sureenough.paths import DERIVED, RAW

_JUMP = RAW / "osf-jum2f"
N_TRAIN_CONCEPTS = 1654
N_VAL_CONCEPTS = 200


@dataclass(frozen=True)
class Concepts:
    test_names: list[str]
    test_things_ids: list[str]
    train_concept_of_image: np.ndarray
    val_concepts: np.ndarray
    cc0_member: dict[str, str]
    categories53: dict[str, set[str]]
    all_names: list[str]
    all_things_ids: list[str]


def _read_tsv(name: str) -> list[dict]:
    with open(_JUMP / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _prefix_index(label: str) -> int:
    return int(label.split("_", 1)[0]) - 1


@lru_cache(maxsize=1)
def load_concepts() -> Concepts:
    meta = json.loads((DERIVED / "image_meta.json").read_text())
    things = _read_tsv("concepts-metadata_things.tsv")
    all_ids = [r["uniqueID"] for r in things]
    all_names = [r["Word"] for r in things]

    test_idx = [_prefix_index(c) for c in meta["test_img_concepts_THINGS"]]
    train_concept = np.array([_prefix_index(c) for c in meta["train_img_concepts"]], dtype=np.int64)

    categories: dict[str, set[str]] = {}
    for r in _read_tsv("category53_long-format.tsv"):
        categories.setdefault(r["category"], set()).add(r["uniqueID"])

    with zipfile.ZipFile(_JUMP / "images_THINGSplus-CC0.zip") as zf:
        cc0 = {n.rsplit("/", 1)[-1].rsplit(".", 1)[0]: n
               for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}

    val = np.sort(np.random.default_rng(0).choice(N_TRAIN_CONCEPTS, N_VAL_CONCEPTS, replace=False))
    return Concepts(
        test_names=[all_names[i] for i in test_idx],
        test_things_ids=[all_ids[i] for i in test_idx],
        train_concept_of_image=train_concept,
        val_concepts=val,
        cc0_member=cc0,
        categories53=categories,
        all_names=all_names,
        all_things_ids=all_ids,
    )
