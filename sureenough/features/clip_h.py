import numpy as np
import torch

from sureenough.models.retrieval import unit_rows
from sureenough.paths import RAW

ATM_DIR = RAW / "hf-atm"


def load_clip_h() -> tuple[np.ndarray, np.ndarray]:
    splits = []
    for split in ("train", "test"):
        blob = torch.load(ATM_DIR / f"ViT-H-14_features_{split}.pt", weights_only=True, map_location="cpu")
        splits.append(unit_rows(blob["img_features"].numpy()))
    return splits[0], splits[1]


def _profiles(x: np.ndarray) -> np.ndarray:
    sims = unit_rows(x) @ unit_rows(x).T
    np.fill_diagonal(sims, np.nan)
    sims = np.where(np.isnan(sims), np.nanmean(sims, axis=1, keepdims=True), sims)
    return (sims - sims.mean(axis=1, keepdims=True)) / sims.std(axis=1, keepdims=True)


def order_agreement(a: np.ndarray, b: np.ndarray) -> float:
    corr = _profiles(a) @ _profiles(b).T
    return float((corr.argmax(axis=1) == np.arange(len(a))).mean())
