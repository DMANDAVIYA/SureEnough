import argparse
import io
import json
import time
import zipfile

import numpy as np
from PIL import Image

from sureenough.data.concepts import load_concepts
from sureenough.features.pe import FINAL_DIM, N_BLOCKS, WIDTH, load_pe, pe_image_features, pe_text_features
from sureenough.paths import DERIVED, RAW

FEAT = DERIVED / "features"
BATCH, SAVE_EVERY = 32, 256


def _members(split: str) -> list[str]:
    meta = json.loads((DERIVED / "image_meta.json").read_text())
    key = "train" if split == "train" else "test"
    folder = "training_images" if split == "train" else "test_images"
    return [f"{folder}/{c}/{f}" for c, f in zip(meta[f"{key}_img_concepts"], meta[f"{key}_img_files"])]


def _open(path, shape, dtype=np.float16):
    mode = "r+" if path.exists() else "w+"
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


def extract(split: str, model, preprocess, limit: int | None = None) -> float:
    members = _members(split)[:limit] if limit else _members(split)
    n = len(members)
    progress = FEAT / f"pe_{split}_progress.json"
    done = json.loads(progress.read_text())["done"] if (progress.exists() and not limit) else 0
    inter = _open(FEAT / f"pe_{split}_inter.npy", (len(_members(split)), N_BLOCKS, 2, WIDTH)) if not limit else None
    final = _open(FEAT / f"pe_{split}_final.npy", (len(_members(split)), FINAL_DIM)) if not limit else None
    zpath = RAW / "osf-y63gw" / ("training_images.zip" if split == "train" else "test_images.zip")
    t0, start = time.time(), done
    with zipfile.ZipFile(zpath) as zf:
        for i in range(done, n, BATCH):
            batch = members[i:i + BATCH]
            images = [Image.open(io.BytesIO(zf.read(m))) for m in batch]
            a, b = pe_image_features(model, preprocess, images)
            if inter is not None:
                inter[i:i + len(batch)], final[i:i + len(batch)] = a, b
            done = i + len(batch)
            if inter is not None and (done % SAVE_EVERY < BATCH or done == n):
                inter.flush(); final.flush()
                progress.write_text(json.dumps({"done": done, "n": n}))
                rate = (done - start) / (time.time() - t0)
                print(f"{split}: {done}/{n}  {rate:.1f} img/s  eta {(n - done) / rate / 60:.0f} min", flush=True)
    return (done - start) / max(time.time() - t0, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0)
    args = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    model, preprocess = load_pe()
    if args.benchmark:
        print(f"benchmark: {extract('test', model, preprocess, limit=args.benchmark):.2f} img/s")
        return
    extract("test", model, preprocess)
    if not (FEAT / "pe_text_final.npy").exists():
        np.save(FEAT / "pe_text_final.npy", pe_text_features(model, load_concepts().all_names))
        print("text features: 1854 names")
    extract("train", model, preprocess)
    print("ALL DONE")


if __name__ == "__main__":
    main()
