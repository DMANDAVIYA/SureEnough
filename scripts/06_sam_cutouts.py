import json
from concurrent.futures import ThreadPoolExecutor

from sureenough.data.concepts import load_concepts
from sureenough.meta_api import COST_SAM, MetaClient, output_text, sam_payload, spark_vision_payload
from sureenough.paths import OUTPUTS
from sureenough.visuals.cc0 import cc0_image, data_url
from sureenough.visuals.sam import cutout, object_mask

VISUALS = OUTPUTS / "visuals"
CUTOUTS = VISUALS / "cutouts"
RAW = VISUALS / "sam_raw"
MIN_AREA = 0.02
DESCRIBE = (
    "Name the single main object in this photo in two to four plain words, "
    "for example 'white plastic spatula'. Reply with only those words."
)
COST_DESCRIBE = 0.003


def cached_segment(client: MetaClient, path, prompt: str, url: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    record = {"prompt": prompt, "response": client.responses(sam_payload(prompt, url), est_cost=COST_SAM)}
    path.write_text(json.dumps(record))
    return record


def usable(record: dict, img):
    mask = object_mask(output_text(record["response"]), img.width, img.height)
    return mask if mask is not None and mask.mean() >= MIN_AREA else None


def process(client: MetaClient, things_id: str, name: str) -> str | None:
    out = CUTOUTS / f"{things_id}.webp"
    if out.exists():
        return None
    img = cc0_image(things_id)
    url = data_url(img)
    mask = usable(cached_segment(client, RAW / f"{things_id}.json", name, url), img)
    if mask is None:
        described = RAW / f"{things_id}_described.json"
        phrase = json.loads(described.read_text())["prompt"] if described.exists() else output_text(
            client.responses(spark_vision_payload("You describe photos tersely.", DESCRIBE, url), est_cost=COST_DESCRIBE)
        ).strip().strip(".'\"")
        mask = usable(cached_segment(client, described, phrase, url), img)
    if mask is None:
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
        fallbacks = sorted(r for r in pool.map(lambda job: process(client, *job), jobs) if r)
    (VISUALS / "sam_fallbacks.json").write_text(json.dumps(fallbacks, indent=1))
    print(f"cutouts: {len(list(CUTOUTS.glob('*.webp')))}/200, plain-tile fallbacks: {len(fallbacks)}, spent ${client.spent():.4f}")


if __name__ == "__main__":
    main()
