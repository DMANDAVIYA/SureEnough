import argparse
import hashlib
import json
import subprocess

import numpy as np

from sureenough.config import PRIMARY, freeze, load_frozen
from sureenough.data.concepts import load_concepts
from sureenough.features.targets import load_targets
from sureenough.models.layout import validation_layout
from sureenough.models.retrieval import cosine_sims
from sureenough.models.store import load_decoder
from sureenough.paths import DERIVED, OUTPUTS, REPO, subject_ids
from sureenough.streams.replay import N_STREAMS, STREAM_SEED, fit_beta, make_streams

MODELS = OUTPUTS / "models"
STREAMS = OUTPUTS / "streams"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_pairs(subject: int, target: str) -> list[tuple[np.ndarray, np.ndarray]]:
    concepts = load_concepts()
    _, galleries = validation_layout(concepts.train_concept_of_image, concepts.val_concepts)
    decoder = load_decoder(MODELS / f"sub-{subject:02d}.pt")
    eeg = np.load(DERIVED / "eeg" / f"sub-{subject:02d}" / "train.npy", mmap_mode="r")
    targets = load_targets(target)[0]
    pairs = []
    for gallery in galleries:
        for k in range(1, 5):
            pred = decoder.embed(np.asarray(eeg[gallery, :k]).mean(axis=1))
            pairs.append((cosine_sims(pred, targets[gallery]), np.arange(len(gallery))))
    return pairs


def freeze_config(target: str, arch: str) -> None:
    betas = {s: fit_beta(validation_pairs(s, target)) for s in subject_ids()}
    cfg = {
        "K": 20,
        "K_secondary": 80,
        "alpha": 0.2,
        "delta": 0.1,
        "m": 3,
        "R": N_STREAMS,
        "lambda_grid": {"start": 1.0, "step": -0.002, "count": 500},
        "n_splits": 1000,
        "split_sizes": [100, 100],
        "seeds": {"streams": STREAM_SEED, "splits": 7, "validation_concepts": 0, "pinned_split": 0},
        "target": target,
        "architecture": arch,
        "beta": {str(s): b for s, b in betas.items()},
        "decoder_sha256": {str(s): _sha256(MODELS / f"sub-{s:02d}.pt") for s in subject_ids()},
        "pvalue": "hoeffding-bentkus",
        "selection": "fixed sequence from lambda=1 downward; deploy the last rejection before the first non-rejection",
    }
    print("frozen:", freeze(cfg))


def assert_committed() -> None:
    rel = PRIMARY.relative_to(REPO).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel, rel.replace(".json", ".sha256")],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    if tracked.returncode != 0 or dirty:
        raise SystemExit("commit configs/primary.json and its .sha256 before touching test streams")


def generate_streams() -> None:
    cfg = load_frozen()
    assert_committed()
    test_targets = load_targets(cfg["target"])[1]
    STREAMS.mkdir(parents=True, exist_ok=True)
    for s in subject_ids():
        out = STREAMS / f"sub-{s:02d}.npy"
        if out.exists():
            continue
        path = MODELS / f"sub-{s:02d}.pt"
        if _sha256(path) != cfg["decoder_sha256"][str(s)]:
            raise SystemExit(f"{path.name} differs from the frozen decoder")
        decoder = load_decoder(path)
        eeg = np.load(DERIVED / "eeg" / f"sub-{s:02d}" / "test.npy")
        probs = make_streams(decoder.embed, eeg, test_targets, s, cfg["beta"][str(s)])
        np.save(out, probs)
        print(f"sub-{s:02d}: streams {probs.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", metavar="TARGET")
    ap.add_argument("--arch", choices=("mlp", "tsconv"), default="mlp")
    ap.add_argument("--streams", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        freeze_config(args.freeze, args.arch)
    if args.streams:
        generate_streams()


if __name__ == "__main__":
    main()
