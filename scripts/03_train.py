import argparse
import json
import time

import numpy as np
import torch

from sureenough.data.concepts import load_concepts
from sureenough.features.targets import load_targets, target_names
from sureenough.models.encoder import MLPEncoder, Standardizer, encode, train_contrastive
from sureenough.models.layout import validation_layout
from sureenough.models.ridge import fit_ridge
from sureenough.models.store import load_decoder, save_decoder
from sureenough.models.tsconv import TSConvEncoder
from sureenough.models.validation import val_top1
from sureenough.paths import DERIVED, OUTPUTS, subject_ids

RIDGE_ALPHAS = (10.0, 100.0, 1e3, 1e4)
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


def build(arch: str, in_dim: int, out_dim: int):
    return MLPEncoder(in_dim, out_dim) if arch == "mlp" else TSConvEncoder(out_dim)


def fit_encoder(z: np.ndarray, y: np.ndarray, train_idx: np.ndarray, galleries: np.ndarray, seed: int, arch: str = "mlp"):
    x = z[train_idx].reshape(len(train_idx), 4, -1)
    torch.manual_seed(seed)
    net = train_contrastive(build(arch, x.shape[-1], y.shape[1]), x, y[train_idx], seed=seed)
    return net, val_top1(lambda v: encode(net, v), z, y, galleries)


def sweep(subjects: list[int], encoder_subjects: list[int], arch: str) -> dict:
    train_idx, galleries = layout()
    names = target_names()
    targets = {name: load_targets(name)[0] for name in [*names, "clip_h"]}
    ridge = {name: {} for name in targets}
    encoder = {}
    for s in subjects:
        _, z = standardized_eeg(s, train_idx)
        for name in targets:
            ridge[name][s] = ridge_top1(z, targets[name], train_idx, galleries)
            print(f"sub-{s:02d} ridge {name}: {ridge[name][s]:.3f}", flush=True)
    ranked = sorted(names, key=lambda n: (-np.mean(list(ridge[n].values())), n != "final"))
    for name in ranked[:2]:
        encoder[name] = {}
        for s in encoder_subjects:
            _, z = standardized_eeg(s, train_idx)
            encoder[name][s] = fit_encoder(z, targets[name], train_idx, galleries, seed=0, arch=arch)[1]
            print(f"sub-{s:02d} {arch} {name}: {encoder[name][s]:.3f}", flush=True)
    chosen = max(encoder, key=lambda n: (np.mean(list(encoder[n].values())), n == "final"))
    return {"subjects": subjects, "encoder_subjects": encoder_subjects, "architecture": arch,
            "ridge": ridge, "encoder": encoder, "chosen": chosen}


def subject_report(s: int, target: str, seeds: list[int], arch: str, train_idx, galleries, y) -> dict:
    standardizer, z = standardized_eeg(s, train_idx)
    report = {}
    for seed in seeds:
        t0 = time.time()
        path = OUTPUTS / "models" / f"sub-{s:02d}{'' if seed == 0 else f'_seed{seed}'}.pt"
        if path.exists():
            decoder = load_decoder(path)
            top1 = val_top1(lambda v: encode(decoder.net, v), z, y, galleries)
        else:
            net, top1 = fit_encoder(z, y, train_idx, galleries, seed, arch)
            save_decoder(path, net, standardizer, target, seed)
        report[str(seed)] = top1
        print(f"sub-{s:02d} seed {seed}: val top-1 {top1:.3f} ({time.time() - t0:.0f}s)", flush=True)
    report["ridge"] = ridge_top1(z, y, train_idx, galleries)
    return report


def train_all(target: str, seeds: list[int], arch: str, subjects: list[int]) -> dict:
    train_idx, galleries = layout()
    y = load_targets(target)[0]
    parts = OUTPUTS / "gate1" / f"{target}_{arch}"
    parts.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        report = subject_report(s, target, seeds, arch, train_idx, galleries, y)
        (parts / f"sub-{s:02d}.json").write_text(json.dumps(report))
    return {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted(parts.glob("sub-*.json"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--encoder-subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--target")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arch", choices=("mlp", "tsconv"), default="mlp")
    ap.add_argument("--train-subjects", type=int, nargs="+", default=subject_ids())
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUTPUTS.mkdir(exist_ok=True)
    if args.sweep:
        result = sweep(args.subjects, args.encoder_subjects, args.arch)
        (OUTPUTS / "sweep.json").write_text(json.dumps(result, indent=1))
        print("chosen target:", result["chosen"])
    if args.all:
        target = args.target or json.loads((OUTPUTS / "sweep.json").read_text())["chosen"]
        report = train_all(target, args.seeds, args.arch, args.train_subjects)
        passed = sum(report[s]["0"] >= GATE1_THRESHOLD for s in report)
        gate1 = {
            "target": target,
            "architecture": args.arch,
            "threshold": GATE1_THRESHOLD,
            "per_subject": report,
            "subjects_passing": passed,
            "passed": passed >= GATE1_MIN_SUBJECTS,
        }
        (OUTPUTS / f"gate1_{target}_{args.arch}.json").write_text(json.dumps(gate1, indent=1))
        print(f"Gate 1 ({target}, {args.arch}): {passed}/{len(report)} subjects >= {GATE1_THRESHOLD}")


if __name__ == "__main__":
    main()
