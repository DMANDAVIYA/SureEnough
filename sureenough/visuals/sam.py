import meta_sam_parser as sam
import numpy as np
from PIL import Image


def response_text(body: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def _mask_records(text: str) -> list:
    if not text.strip():
        return []
    parser = sam.image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(sam.CompletedOutcome()).result
    return [record for record in result.records if record.kind == "mask"]


def best_mask(text: str, width: int, height: int) -> np.ndarray | None:
    records = _mask_records(text)
    if not records:
        return None
    record = max(records, key=lambda r: r.confidence or 0.0)
    m = record.mask
    raster = np.frombuffer(sam.decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width).astype(bool)
    top, left = record.bounds.top, record.bounds.left
    rows, cols = min(m.height, height - top), min(m.width, width - left)
    frame = np.zeros((height, width), dtype=bool)
    frame[top:top + rows, left:left + cols] = raster[:rows, :cols]
    return frame


def cutout(img: Image.Image, mask: np.ndarray, pad: float = 0.08) -> Image.Image:
    ys, xs = np.nonzero(mask)
    margin = round(pad * max(np.ptp(xs) + 1, np.ptp(ys) + 1))
    box = (
        max(int(xs.min()) - margin, 0),
        max(int(ys.min()) - margin, 0),
        min(int(xs.max()) + 1 + margin, img.width),
        min(int(ys.max()) + 1 + margin, img.height),
    )
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    return rgba.crop(box)
