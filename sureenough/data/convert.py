import json
import zipfile
from pathlib import Path

import numpy as np

from sureenough.data.safe_npy import load_pickled_npy

_MEMBERS = {"test": "preprocessed_eeg_test.npy", "train": "preprocessed_eeg_training.npy"}


def convert_subject(zip_path: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes, meta = {}, {}
    with zipfile.ZipFile(zip_path) as zf:
        for split, suffix in _MEMBERS.items():
            member = next(n for n in zf.namelist() if n.endswith(suffix))
            with zf.open(member) as f:
                d = load_pickled_npy(f)
            arr = np.asarray(d.pop("preprocessed_eeg_data")).astype(np.float32)
            np.save(out_dir / f"{split}.npy", arr)
            shapes[split] = tuple(arr.shape)
            meta = {"ch_names": [str(c) for c in d["ch_names"]],
                    "times": [float(t) for t in np.asarray(d["times"])]}
            del arr, d
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return shapes


def convert_image_metadata(npy_path: Path, out_json: Path) -> None:
    with open(npy_path, "rb") as f:
        d = load_pickled_npy(f)
    out = {k: [str(x) for x in np.asarray(v).tolist()] for k, v in d.items()}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out))
