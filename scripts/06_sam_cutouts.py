import json
from concurrent.futures import ThreadPoolExecutor

from sureenough.data.concepts import load_concepts
from sureenough.meta_api import COST_SAM, MetaClient, output_text, sam_payload
from sureenough.paths import OUTPUTS
from sureenough.visuals.cc0 import cc0_image, data_url
from sureenough.visuals.sam import best_mask, cutout

VISUALS = OUTPUTS / "visuals"
CUTOUTS = VISUALS / "cutouts"
RAW = VISUALS / "sam_raw"
MIN_AREA = 0.02


def process(client: MetaClient, things_id: str, name: str) -> str | None:
    out = CUTOUTS / f"{things_id}.webp"
    if out.exists():
        return None
    img = cc0_image(things_id)
    body = client.responses(sam_payload(name, data_url(img)), est_cost=COST_SAM)
    (RAW / f"{things_id}.json").write_text(json.dumps(body))
    mask = best_mask(output_text(body), img.width, img.height)
    if mask is None or mask.mean() < MIN_AREA:
        img.convert("RGBA").save(out, quality=90)
        return things_id
    cutout(img, mask).save(out, quality=90)
    return None


def main() -> None:
    concepts = load_concepts()
    CUTOUTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    client = MetaClient()
    jobs = list(zip(concepts.test_things_ids, concepts.test_names))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda job: process(client, *job), jobs))
    fallbacks_path = VISUALS / "sam_fallbacks.json"
    previous = json.loads(fallbacks_path.read_text()) if fallbacks_path.exists() else []
    fallbacks = sorted(set(previous) | {r for r in results if r})
    fallbacks_path.write_text(json.dumps(fallbacks, indent=1))
    print(f"cutouts: {len(list(CUTOUTS.glob('*.webp')))}/200, plain-tile fallbacks: {len(fallbacks)}, spent ${client.spent():.4f}")


if __name__ == "__main__":
    main()
