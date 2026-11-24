"""Train the AquaWatch AI U-Net on synthetic water scenes.

Example
-------
    python train_unet.py --epochs 4 --size 128 --steps-per-epoch 40

The script writes ``models/water_unet.pt`` (architecture config + weights +
validation metrics) which ``app.py`` loads automatically. Swap the dataset for
real Sentinel-2 chips and the rest of the pipeline keeps working.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import synthetic
from unet_model import build_unet, save_checkpoint


class SyntheticWaterDataset(torch.utils.data.Dataset):
    """Water scenes generated on the fly, deterministic per index."""

    def __init__(self, size: int = 128, seed: int = 0, length: int = 10_000) -> None:
        self.size = size
        self.seed = seed
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene_seed = self.seed + index
        rng = np.random.default_rng(5000 + scene_seed)
        shrink = float(rng.uniform(0.0, 0.55))
        intrusion = bool(rng.random() < 0.30)
        image, mask = synthetic.make_scene(
            seed=scene_seed, size=self.size, shrink=shrink, intrusion=intrusion
        )

        if rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            image, mask = image[::-1], mask[::-1]

        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        target = (mask > 127).astype(np.float32)[None]
        return torch.from_numpy(image), torch.from_numpy(target)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return 1.0 - ((2 * intersection + eps) / (denominator + eps)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    intersection = union = 0.0
    dice_sum = 0.0
    batches = 0
    for step, (images, targets) in enumerate(loader):
        if step >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        predictions = (torch.sigmoid(model(images)) >= 0.5).float()
        intersection += float((predictions * targets).sum().item())
        union += float(((predictions + targets) > 0).float().sum().item())
        dice_sum += float(
            (2 * (predictions * targets).sum().item())
            / max(predictions.sum().item() + targets.sum().item(), 1.0)
        )
        batches += 1
    model.train()
    iou = intersection / union if union else 0.0
    return {"val_iou": iou, "val_dice": dice_sum / max(batches, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AquaWatch AI U-Net.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--val-steps", type=int, default=8)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/water_unet.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)

    train_data = SyntheticWaterDataset(size=args.size, seed=args.seed)
    val_data = SyntheticWaterDataset(size=args.size, seed=args.seed + 500_000)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=0)

    model = build_unet(in_channels=3, base=args.base, depth=args.depth).to(device)
    print(
        f"device={device} parameters={model.parameter_count():,} "
        f"images={args.size}x{args.size} epochs={args.epochs}"
    )

    bce = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    model.train()
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        running = 0.0
        steps = 0
        for step, (images, targets) in enumerate(train_loader):
            if step >= args.steps_per_epoch:
                break
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = bce(logits, targets) + dice_loss(logits, targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += float(loss.item())
            steps += 1
            if steps % 10 == 0:
                print(f"  epoch {epoch} step {steps}/{args.steps_per_epoch} loss={running / steps:.4f}")
        scheduler.step()

        metrics = evaluate(model, val_loader, device, args.val_steps)
        elapsed = time.time() - started
        print(
            f"epoch {epoch}/{args.epochs} loss={running / max(steps, 1):.4f} "
            f"val_iou={metrics['val_iou']:.4f} val_dice={metrics['val_dice']:.4f} "
            f"({elapsed:.1f}s)"
        )

    path = save_checkpoint(
        args.out,
        model,
        extra={
            "trained_on": "synthetic scenes (synthetic.make_scene)",
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "train_size": args.size,
            **metrics,
        },
    )
    print(f"saved checkpoint -> {Path(path).resolve()}")


if __name__ == "__main__":
    main()