import numpy as np
import torch

from sureenough.paths import use_project_hf_cache

PE_HUB_ID = "hf-hub:timm/PE-Core-B-16"
N_BLOCKS, WIDTH, FINAL_DIM = 12, 768, 1024


def load_pe(threads: int = 10):
    use_project_hf_cache()
    import open_clip

    torch.set_num_threads(threads)
    model, _, preprocess = open_clip.create_model_and_transforms(PE_HUB_ID)
    return model.eval(), preprocess


@torch.no_grad()
def pe_image_features(model, preprocess, images) -> tuple[np.ndarray, np.ndarray]:
    x = torch.stack([preprocess(im.convert("RGB")) for im in images])
    trunk = model.visual.trunk
    last, inter = trunk.forward_intermediates(
        x, indices=N_BLOCKS, return_prefix_tokens=True, output_fmt="NLC"
    )
    out = np.empty((len(images), N_BLOCKS, 2, WIDTH), dtype=np.float16)
    for layer, (patches, prefix) in enumerate(inter):
        out[:, layer, 0] = prefix[:, 0].numpy()
        out[:, layer, 1] = patches.mean(dim=1).numpy()
    final = model.visual.head(trunk.forward_head(last))
    return out, final.numpy().astype(np.float16)


@torch.no_grad()
def pe_text_features(model, names: list[str]) -> np.ndarray:
    import open_clip

    tokens = open_clip.get_tokenizer(PE_HUB_ID)([f"a photo of a {n}" for n in names])
    return model.encode_text(tokens).numpy().astype(np.float16)
