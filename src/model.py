"""Sprint 3: residual CNN policy network.

Input : (B, 18, 8, 8)  float32 planes from encode.py
Output: (B, 4168)      raw scores (logits), one per move index from moves.py
"""
import torch
import torch.nn as nn

from encode import NUM_PLANES
from moves import NUM_MOVES

PRESETS = {
    "small": dict(channels=64, blocks=4),     # for CPU debugging / quick experiments
    "base": dict(channels=128, blocks=6),     # the real model (train on a GPU)
}


class ResBlock(nn.Module):
    """x -> conv-BN-ReLU -> conv-BN -> (+ x) -> ReLU"""

    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)             # the skip connection


class PolicyNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 6, head_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
            nn.Flatten(),                                   # (B, head_channels * 64)
            nn.Linear(head_channels * 64, NUM_MOVES),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def build(preset: str = "base") -> PolicyNet:
    return PolicyNet(**PRESETS[preset])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())