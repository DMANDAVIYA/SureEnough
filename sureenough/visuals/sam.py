import meta_sam_parser as sam
import numpy as np
from PIL import Image


def _mask_records(text: str) -> list:
    if not text.strip():
        return []
    parser = sam.image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(sam.CompletedOutcome()).result
    return [record for record in result.records if record.kind == "mask"]


def _placed(record, width: int, height: int) -> np.ndarray:
    m = record.mask
    raster = np.frombuffer(sam.decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width).astype(bool)
    top, left = record.bounds.top, record.bounds.left
    rows, cols = min(m.height, height - top), min(m.width, width - left)
    frame = np.zeros((height, width), dtype=bool)
    frame[top:top + rows, left:left + cols] = raster[:rows, :cols]
    return frame


def object_mask(text: str, width: int, height: int, relative_confidence: float = 0.5) -> np.ndarray | None:
    records = _mask_records(text)
    if not records:
        return None
    floor = relative_confidence * max(r.confidence or 0.0 for r in records)
    confident = [r for r in records if (r.confidence or 0.0) >= floor]
    return np.logical_or.reduce([_placed(r, width, height) for r in confident])


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
