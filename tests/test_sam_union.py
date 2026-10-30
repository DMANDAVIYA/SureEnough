import json
from pathlib import Path

from sureenough.meta_api import output_text, spark_vision_payload
from sureenough.visuals.sam import object_mask

SAM_FIXTURE = Path(__file__).parent / "fixtures" / "api" / "sam.json"


def _first() -> str:
    return output_text(json.loads(SAM_FIXTURE.read_text(encoding="utf-8")))


def _two_records(second_confidence: float) -> str:
    first = _first()
    second = (
        "1" + first.removeprefix("<0f>0")
        .replace("x1=85;y1=126;x2=653;y2=564", "x1=0;y1=0;x2=568;y2=438")
        .replace("c=0.9726027250289917", f"c={second_confidence}")
    )
    return first.rstrip() + "," + second


def test_object_mask_unions_confident_records():
    text = _two_records(0.9)
    single = object_mask(_first(), 768, 768)
    union = object_mask(text, 768, 768)
    assert union.sum() > single.sum()
    assert union[0, 0] or union[:439, :569].any()


def test_object_mask_drops_weak_records():
    text = _two_records(0.2)
    assert object_mask(text, 768, 768).sum() == object_mask(_first(), 768, 768).sum()


def test_object_mask_none_without_records():
    assert object_mask("", 10, 10) is None


def test_spark_vision_payload_carries_image_and_no_storage():
    payload = spark_vision_payload("sys", "Name it.", "data:image/jpeg;base64,AAAA")
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Name it."}
    assert content[1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"}
    assert payload["store"] is False and payload["model"] == "muse-spark-1.3"
