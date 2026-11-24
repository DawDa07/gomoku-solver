"""Convolutional policy and value network.

The trunk is fully convolutional, with residual blocks, so weights trained on a
9×9 board still run on the 15×15 board. The stack sees the whole 15×15 board.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine.board import Board, opponent

CHANNELS = 48
DEPTH = 8
CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "learned.pt"


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = CHANNELS, depth: int = DEPTH):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList(_Block(channels) for _ in range(depth // 2))
        self.policy = nn.Conv2d(channels, 1, 1)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.value = nn.Linear(4, 1)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.stem(boards)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.policy(hidden).flatten(1)
        pooled = self.value_conv(hidden).flatten(1)
        return logits, torch.tanh(self.value(pooled).squeeze(-1))


class _Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.out = nn.ReLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(hidden + self.conv(hidden))


def encode(board: Board, side: int) -> np.ndarray:
    """Three planes, from the side about to move: own stones, opponent, ones."""
    size = board.size
    cells = np.asarray(board.cells, dtype=np.int8).reshape(size, size)
    planes = np.empty((3, size, size), np.float32)
    planes[0] = cells == side
    planes[1] = cells == opponent(side)
    planes[2] = 1.0
    return planes


def load_checkpoint(device: torch.device, path: Path = CHECKPOINT) -> PolicyValueNet | None:
    if not path.is_file():
        return None
    saved = torch.load(path, map_location=device, weights_only=False)
    net = PolicyValueNet(saved.get("channels", CHANNELS), saved.get("depth", DEPTH))
    net.load_state_dict(saved["state_dict"])
    net.to(device)
    net.eval()
    return net


def save_checkpoint(net: PolicyValueNet, path: Path = CHECKPOINT, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "channels": CHANNELS,
        "depth": DEPTH,
    }
    payload.update(extra)
    torch.save(payload, path)
