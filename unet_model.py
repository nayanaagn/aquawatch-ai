"""U-Net segmentation network used by AquaWatch AI.

The network is deliberately small (a few hundred thousand parameters) so the
demo checkpoint can be trained on a laptop CPU in a couple of minutes, while
still learning to separate water from land in satellite crops.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool then two convolutions: the encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Transposed convolution, skip connection, two convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard U-Net with configurable width and depth.

    ``depth`` is the number of down-sampling stages, so inputs must be
    divisible by ``2 ** depth`` (the detector pads images for you).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        channels = [base * (2**i) for i in range(depth + 1)]
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base": base,
            "depth": depth,
            "channels": channels,
        }
        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(
            Down(channels[i], channels[i + 1]) for i in range(depth)
        )
        # Built deepest-first so `ups` lines up with the encoder skips.
        self.ups = nn.ModuleList(
            Up(channels[i + 1], channels[i], channels[i]) for i in range(depth - 1, -1, -1)
        )
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        x = features.pop()  # bottleneck
        for up in self.ups:
            x = up(x, features.pop())
        return self.outc(x)

    @property
    def size_multiple(self) -> int:
        return 2 ** int(self.config["depth"])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(in_channels: int = 3, base: int = 16, depth: int = 4) -> UNet:
    return UNet(in_channels=in_channels, out_channels=1, base=base, depth=depth)


def save_checkpoint(path: str | Path, model: UNet, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": dict(model.config),
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "framework": f"pytorch {torch.__version__}",
            **(extra or {}),
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[UNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = payload["config"]
    model = build_unet(
        in_channels=int(config.get("in_channels", 3)),
        base=int(config.get("base", 16)),
        depth=int(config.get("depth", 4)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, dict(payload.get("meta", {}))