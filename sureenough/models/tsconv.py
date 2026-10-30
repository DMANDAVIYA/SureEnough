import torch
import torch.nn.functional as F
from torch import nn

KERNEL, POOL, STRIDE = 10, 20, 4


class TSConvEncoder(nn.Module):
    def __init__(self, out_dim: int, channels: int = 17, times: int = 100, filters: int = 40, p_drop: float = 0.5):
        super().__init__()
        self.config = {"out_dim": out_dim, "channels": channels, "times": times, "filters": filters, "p_drop": p_drop}
        self.channels, self.times = channels, times
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, (1, KERNEL)),
            nn.AvgPool2d((1, POOL), (1, STRIDE)),
            nn.BatchNorm2d(filters),
            nn.ELU(),
            nn.Conv2d(filters, filters, (channels, 1)),
            nn.BatchNorm2d(filters),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Conv2d(filters, filters, 1),
            nn.Flatten(),
        )
        flat = filters * ((times - KERNEL + 1 - POOL) // STRIDE + 1)
        self.project = nn.Linear(flat, out_dim)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(out_dim, out_dim), nn.Dropout(p_drop))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project(self.features(x.reshape(-1, 1, self.channels, self.times)))
        return F.normalize(self.norm(h + self.residual(h)), dim=-1)
