import numpy as np
from PIL import Image

GREY = 128.0
FILL = 0.9


def _fit(member: Image.Image, size: int) -> np.ndarray:
    scale = FILL * size / max(member.size)
    resized = member.resize((max(1, round(member.width * scale)), max(1, round(member.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas, dtype=np.float64) / 255.0


def composite(members: list[Image.Image], weights: list[float], size: int = 512) -> Image.Image:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.zeros((size, size, 3))
    for member, weight in zip(members, w):
        layer = _fit(member.convert("RGBA"), size)
        alpha = layer[..., 3:]
        out += weight * (layer[..., :3] * 255.0 * alpha + GREY * (1.0 - alpha))
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))
