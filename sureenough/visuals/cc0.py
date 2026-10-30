import base64
import io
import zipfile

from PIL import Image

from sureenough.data.concepts import load_concepts
from sureenough.paths import RAW

CC0_ZIP = RAW / "osf-jum2f" / "images_THINGSplus-CC0.zip"


def cc0_image(things_id: str, max_side: int = 768) -> Image.Image:
    with zipfile.ZipFile(CC0_ZIP) as zf:
        img = Image.open(io.BytesIO(zf.read(load_concepts().cc0_member[things_id]))).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
