import numpy as np

from sureenough.paths import DERIVED

FEATURES = DERIVED / "features"
SWEEP_LAYERS = range(6, 12)
_KINDS = {"cls": 0, "mean": 1}


def target_names() -> list[str]:
    return ["final"] + [f"L{layer}_{kind}" for layer in SWEEP_LAYERS for kind in _KINDS]


def parse_target_name(name: str) -> tuple[int, int] | None:
    if name == "final":
        return None
    layer, _, kind = name.removeprefix("L").partition("_")
    if not layer.isdigit() or int(layer) not in range(12) or kind not in _KINDS:
        raise ValueError(f"unknown target {name!r}")
    return int(layer), _KINDS[kind]


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def load_targets(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "clip_h":
        from sureenough.features.clip_h import load_clip_h

        return load_clip_h()
    parsed = parse_target_name(name)
    splits = []
    for split in ("train", "test"):
        if parsed is None:
            splits.append(_unit(np.load(FEATURES / f"pe_{split}_final.npy", mmap_mode="r")))
        else:
            inter = np.load(FEATURES / f"pe_{split}_inter.npy", mmap_mode="r")
            splits.append(_unit(inter[:, parsed[0], parsed[1]]))
    return splits[0], splits[1]
