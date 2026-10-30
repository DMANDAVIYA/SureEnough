import io
import json
import zipfile

import numpy as np
from PIL import Image

from sureenough.analysis.h5a import h5a
from sureenough.analysis.splits import split
from sureenough.config import load_frozen
from sureenough.data.concepts import load_concepts
from sureenough.demo import demo_subjects, load_streams, pinned_lambda
from sureenough.features.dino import dino_embed, load_dino
from sureenough.paths import DERIVED, OUTPUTS, RAW
from sureenough.visuals.composite import composite
from sureenough.visuals.conditions import CONDITIONS, condition_members

CUTOUTS = OUTPUTS / "visuals" / "cutouts"
IMPRESSIONS = OUTPUTS / "visuals" / "impressions"
MAX_COMPOSITE_MEMBERS = 12


def stimulus(zf: zipfile.ZipFile, meta: dict, image: int) -> Image.Image:
    member = f"test_images/{meta['test_img_concepts'][image]}/{meta['test_img_files'][image]}"
    return Image.open(io.BytesIO(zf.read(member))).convert("RGB")


def composite_of(concepts, members: list[int], weights: list[float]) -> Image.Image:
    shown = slice(0, MAX_COMPOSITE_MEMBERS)
    return composite([Image.open(CUTOUTS / f"{concepts.test_things_ids[j]}.webp") for j in members[shown]], weights[shown])


def score_subject(model, zf, meta, concepts, cfg, subject: int, sims: dict) -> None:
    probs = load_streams(subject)
    lam = pinned_lambda(subject)
    _, eval_idx = split(subject, 0, len(probs))
    for image in map(int, eval_idx):
        conds = condition_members(probs, image, lam, cfg["K"], cfg["m"])
        if conds["top1"][0][0] == image:
            continue
        truth = dino_embed(model, [stimulus(zf, meta, image)])[0]
        for cond in CONDITIONS:
            sims[f"composite_{cond}"].append(float(dino_embed(model, [composite_of(concepts, *conds[cond])])[0] @ truth))
            files = sorted((IMPRESSIONS / f"sub-{subject:02d}").glob(f"{image:03d}_{cond}_*.webp"))
            if files:
                sims[f"impression_{cond}"].append(float((dino_embed(model, [Image.open(f) for f in files]) @ truth).mean()))


def main() -> None:
    cfg = load_frozen()
    concepts = load_concepts()
    meta = json.loads((DERIVED / "image_meta.json").read_text())
    model = load_dino()
    sims = {f"{kind}_{cond}": [] for kind in ("composite", "impression") for cond in CONDITIONS}
    with zipfile.ZipFile(RAW / "osf-y63gw" / "test_images.zip") as zf:
        for subject in demo_subjects():
            score_subject(model, zf, meta, concepts, cfg, subject, sims)
    impressions_paired = 0 < len(sims["impression_top1"]) == len(sims["impression_hedged"])
    out = h5a(np.array(sims["composite_hedged"]), np.array(sims["composite_top1"])) | {
        "visual": "SAM 3.1 composite",
        "impression_secondary": h5a(np.array(sims["impression_hedged"]), np.array(sims["impression_top1"])) if impressions_paired else None,
        "h5b_means": {k: float(np.mean(v)) for k, v in sims.items() if v},
    }
    (OUTPUTS / "h5a.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
