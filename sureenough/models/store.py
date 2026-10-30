from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sureenough.models.encoder import MLPEncoder, Standardizer, encode
from sureenough.models.tsconv import TSConvEncoder

ARCHITECTURES = {cls.__name__: cls for cls in (MLPEncoder, TSConvEncoder)}


@dataclass(frozen=True)
class Decoder:
    net: nn.Module
    standardizer: Standardizer
    target: str
    seed: int

    def embed(self, eeg: np.ndarray) -> np.ndarray:
        return encode(self.net, self.standardizer.transform(eeg))


def save_decoder(path: Path, net: nn.Module, standardizer: Standardizer, target: str, seed: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "architecture": type(net).__name__,
            "config": net.config,
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
            "target": target,
            "seed": seed,
        },
        path,
    )


def load_decoder(path: Path) -> Decoder:
    blob = torch.load(path, weights_only=True)
    net = ARCHITECTURES[blob["architecture"]](**blob["config"])
    net.load_state_dict(blob["state_dict"])
    standardizer = Standardizer(np.asarray(blob["mean"], np.float32), np.asarray(blob["std"], np.float32))
    return Decoder(net.eval(), standardizer, blob["target"], blob["seed"])
