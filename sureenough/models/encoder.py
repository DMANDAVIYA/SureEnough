import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "Standardizer":
        mean = train.mean(axis=(0, 1, 3), dtype=np.float64)
        std = train.std(axis=(0, 1, 3), dtype=np.float64)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def sample_repeat_average(x: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    batch, repeats = x.shape[:2]
    counts = torch.randint(1, repeats + 1, (batch,), generator=gen)
    order = torch.rand(batch, repeats, generator=gen).argsort(dim=1).argsort(dim=1)
    mask = (order < counts[:, None]).to(x.dtype)
    return (x * mask[..., None]).sum(dim=1) / counts[:, None].to(x.dtype), counts


def train_mlp(
    x_rep: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch: int = 512,
    lr: float = 1e-3,
    wd: float = 0.01,
    seed: int = 0,
    hidden: int = 2048,
    p_drop: float = 0.3,
) -> MLPEncoder:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(x_rep, dtype=np.float32))
    targets = F.normalize(torch.from_numpy(np.asarray(y, dtype=np.float32)), dim=1)
    model = MLPEncoder(x.shape[-1], targets.shape[1], hidden, p_drop)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    opt = torch.optim.AdamW([*model.parameters(), logit_scale], lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(x) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=gen).split(batch):
            avg, _ = sample_repeat_average(x[idx], gen)
            logits = logit_scale.exp().clamp(max=100) * model(avg) @ targets[idx].T
            labels = torch.arange(len(idx))
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def encode(model: MLPEncoder, x: np.ndarray, chunk: int = 4096) -> np.ndarray:
    model.eval()
    flat = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).reshape(len(x), -1))
    return torch.cat([model(part) for part in flat.split(chunk)]).numpy()
