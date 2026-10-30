from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sureenough.models.encoder import MLPEncoder, Standardizer, encode


@dataclass(frozen=True)
class Decoder:
    net: MLPEncoder
    standardizer: Standardizer
    target: str
    seed: int

    def embed(self, eeg: np.ndarray) -> np.ndarray:
        return encode(self.net, self.standardizer.transform(eeg))


def save_decoder(path: Path, net: MLPEncoder, standardizer: Standardizer, target: str, seed: int) -> None:
    first, last = net.net[0], net.net[-1]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "dims": [first.in_features, last.out_features, first.out_features],
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
            "target": target,
            "seed": seed,
        },
        path,
    )


def load_decoder(path: Path) -> Decoder:
    blob = torch.load(path, weights_only=True)
    in_dim, out_dim, hidden = blob["dims"]
    net = MLPEncoder(in_dim, out_dim, hidden)
    net.load_state_dict(blob["state_dict"])
    standardizer = Standardizer(np.asarray(blob["mean"], np.float32), np.asarray(blob["std"], np.float32))
    return Decoder(net.eval(), standardizer, blob["target"], blob["seed"])
