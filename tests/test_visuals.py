import json
from pathlib import Path

import numpy as np
from PIL import Image

from sureenough.visuals.composite import composite
from sureenough.visuals.sam import best_mask, cutout, response_text

SAM_FIXTURE = Path(__file__).parent / "fixtures" / "api" / "sam.json"


def _sam_text() -> str:
    return response_text(json.loads(SAM_FIXTURE.read_text(encoding="utf-8")))


def test_best_mask_places_recorded_mask_inside_its_box():
    mask = best_mask(_sam_text(), width=768, height=768)
    assert mask.shape == (768, 768) and mask.dtype == bool
    ys, xs = np.nonzero(mask)
    assert xs.min() >= 85 and xs.max() <= 653 and ys.min() >= 126 and ys.max() <= 564
    assert 0.05 < mask.mean() < 0.5


def test_best_mask_is_none_without_records():
    assert best_mask("", width=10, height=10) is None


def test_cutout_is_transparent_outside_mask_and_cropped():
    img = Image.new("RGB", (100, 100), (200, 10, 10))
    mask = np.zeros((100, 100), bool)
    mask[40:60, 30:50] = True
    out = cutout(img, mask, pad=0.1)
    assert out.mode == "RGBA"
    assert out.size == (24, 24)
    alpha = np.asarray(out)[..., 3]
    assert alpha[0, 0] == 0 and alpha[12, 12] == 255


def _square(color):
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (20, 20), color), (10, 10))
    return img


def test_single_member_composite_is_cutout_on_grey():
    out = composite([_square((255, 0, 0, 255))], [1.0], size=64)
    arr = np.asarray(out)
    assert out.mode == "RGB" and out.size == (64, 64)
    assert tuple(arr[0, 0]) == (128, 128, 128)
    assert tuple(arr[32, 32]) == (255, 0, 0)


def test_composite_normalises_weights():
    members = [_square((255, 0, 0, 255)), _square((0, 0, 255, 255))]
    a = np.asarray(composite(members, [1.0, 1.0], size=64))
    b = np.asarray(composite(members, [0.5, 0.5], size=64))
    assert np.array_equal(a, b)
    assert tuple(a[32, 32]) == (128, 0, 128)
