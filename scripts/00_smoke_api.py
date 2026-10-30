import base64
import json

from sureenough.meta_api import (
    COST_IMAGE,
    COST_SAM,
    MetaAPIError,
    MetaClient,
    image_edit_payload,
    sam_payload,
    spark_payload,
)
from sureenough.paths import OUTPUTS, REPO
from sureenough.visuals.cc0 import cc0_image, data_url

FIXTURES = REPO / "tests" / "fixtures" / "api"
SMOKE = OUTPUTS / "smoke"
CONCEPT = "antelope"


def save(name: str, body: dict) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.json").write_text(json.dumps(body, indent=1))


def main() -> None:
    client = MetaClient()
    SMOKE.mkdir(parents=True, exist_ok=True)
    image_url = data_url(cc0_image(CONCEPT))

    spark = client.responses(spark_payload("Reply with exactly one word.", "Say OK."), est_cost=0.001)
    save("spark", spark)
    print("spark ok")

    sam = client.responses(sam_payload(CONCEPT, image_url), est_cost=COST_SAM)
    save("sam", sam)
    print("sam ok")

    flags = {"muse_image": False, "error": None}
    try:
        prompt = f"An artist's impression of an {CONCEPT}. Simple, centred, neutral grey background."
        image = client.image_edit(image_edit_payload(prompt, [image_url], n=1), est_cost=COST_IMAGE)
        (SMOKE / "impression.webp").write_bytes(base64.b64decode(image["data"][0]["b64_json"]))
        image["data"] = [{"b64_json": "<omitted>"}]
        save("image", image)
        flags["muse_image"] = True
        print("muse image ok")
    except MetaAPIError as err:
        flags["error"] = str(err)
        print("muse image unavailable:", err)
    (OUTPUTS / "flags.json").write_text(json.dumps(flags, indent=1))
    print(f"spent ${client.spent():.4f}")


if __name__ == "__main__":
    main()
