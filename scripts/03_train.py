import argparse
import json
import time

import numpy as np

from sureenough.data.concepts import load_concepts
from sureenough.features.targets import load_targets, target_names
from sureenough.models.encoder import Standardizer, encode, train_mlp
from sureenough.models.layout import validation_layout
from sureenough.models.ridge import fit_ridge
from sureenough.models.store import save_decoder
from sureenough.models.validation import val_top1
from sureenough.paths import DERIVED, OUTPUTS, subject_ids

RIDGE_ALPHAS = (1e3, 1e4, 1e5, 1e6)
GATE1_THRESHOLD = 0.15
GATE1_MIN_SUBJECTS = 8


def layout() -> tuple[np.ndarray, np.ndarray]:
    concepts = load_concepts()
    train_idx, galleries = validation_layout(concepts.train_concept_of_image, concepts.val_concepts)
    assert not np.isin(concepts.train_concept_of_image[train_idx], concepts.val_concepts).any()
    return train_idx, galleries


def standardized_eeg(subject: int, train_idx: np.ndarray) -> tuple[Standardizer, np.ndarray]:
    eeg = np.load(DERIVED / "eeg" / f"sub-{subject:02d}" / "train.npy")
    standardizer = Standardizer.fit(eeg[train_idx])
    return standardizer, standardizer.transform(eeg)


def ridge_top1(z: np.ndarray, y: np.ndarray, train_idx: np.ndarray, galleries: np.ndarray) -> float:
    x = z[train_idx].mean(axis=1).reshape(len(train_idx), -1)
    return max(val_top1(fit_ridge(x, y[train_idx], a).predict, z, y, galleries) for a in RIDGE_ALPHAS)


def mlp_fit(z: np.ndarray, y: np.ndarray, train_idx: np.ndarray, galleries: np.ndarray, seed: int):
    net = train_mlp(z[train_idx].reshape(len(train_idx), 4, -1), y[train_idx], seed=seed)
    return net, val_top1(lambda x: encode(net, x), z, y, galleries)


def sweep(subjects: list[int]) -> dict:
    train_idx, galleries = layout()
    names = target_names()
    targets = {name: load_targets(name)[0] for name in names}
    ridge = {name: {} for name in names}
    mlp = {}
    for s in subjects:
        _, z = standardized_eeg(s, train_idx)
        for name in names:
            ridge[name][s] = ridge_top1(z, targets[name], train_idx, galleries)
            print(f"sub-{s:02d} ridge {name}: {ridge[name][s]:.3f}", flush=True)
    ranked = sorted(names, key=lambda n: (-np.mean(list(ridge[n].values())), n != "final"))
    for name in ranked[:2]:
        mlp[name] = {}
        for s in subjects:
            _, z = standardized_eeg(s, train_idx)
            mlp[name][s] = mlp_fit(z, targets[name], train_idx, galleries, seed=0)[1]
            print(f"sub-{s:02d} mlp {name}: {mlp[name][s]:.3f}", flush=True)
    chosen = max(mlp, key=lambda n: (np.mean(list(mlp[n].values())), n == "final"))
    return {"subjects": subjects, "ridge": ridge, "mlp": mlp, "chosen": chosen}


def train_all(target: str, seeds: list[int]) -> dict:
    train_idx, galleries = layout()
    y = load_targets(target)[0]
    report = {}
    for s in subject_ids():
        standardizer, z = standardized_eeg(s, train_idx)
        for seed in seeds:
            t0 = time.time()
            net, top1 = mlp_fit(z, y, train_idx, galleries, seed)
            suffix = "" if seed == 0 else f"_seed{seed}"
            save_decoder(OUTPUTS / "models" / f"sub-{s:02d}{suffix}.pt", net, standardizer, target, seed)
            report.setdefault(s, {})[seed] = top1
            print(f"sub-{s:02d} seed {seed}: val top-1 {top1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        report[s]["ridge"] = ridge_top1(z, y, train_idx, galleries)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--target")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    if args.sweep:
        result = sweep(args.subjects)
        (OUTPUTS / "sweep.json").write_text(json.dumps(result, indent=1))
        print("chosen target:", result["chosen"])
    if args.all:
        target = args.target or json.loads((OUTPUTS / "sweep.json").read_text())["chosen"]
        report = train_all(target, args.seeds)
        passed = sum(report[s][0] >= GATE1_THRESHOLD for s in report)
        gate1 = {
            "target": target,
            "threshold": GATE1_THRESHOLD,
            "per_subject": report,
            "subjects_passing": passed,
            "passed": passed >= GATE1_MIN_SUBJECTS,
        }
        (OUTPUTS / f"gate1_{target}.json").write_text(json.dumps(gate1, indent=1))
        print(f"Gate 1 ({target}): {passed}/10 subjects >= {GATE1_THRESHOLD}")


if __name__ == "__main__":
    main()
