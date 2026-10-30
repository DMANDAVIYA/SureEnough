import numpy as np
import torch
from torchvision import transforms

from sureenough.paths import use_project_hf_cache

DINO_ID = "vit_large_patch14_dinov2.lvd142m"
SIZE = 224
_PREPROCESS = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_dino(threads: int = 10):
    use_project_hf_cache()
    import timm

    torch.set_num_threads(threads)
    return timm.create_model(DINO_ID, pretrained=True, img_size=SIZE, num_classes=0).eval()


@torch.no_grad()
def dino_embed(model, images, batch: int = 16) -> np.ndarray:
    chunks = []
    for start in range(0, len(images), batch):
        x = torch.stack([_PREPROCESS(im.convert("RGB")) for im in images[start:start + batch]])
        chunks.append(torch.nn.functional.normalize(model(x), dim=-1))
    return torch.cat(chunks).numpy()
