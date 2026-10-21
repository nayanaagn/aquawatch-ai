"""
Starter U-Net training script.

Expected data layout:
data/patches/images/*.npy  -> float arrays shaped (C, H, W)
data/patches/masks/*.npy   -> binary arrays shaped (H, W) or (1, H, W)

Each image patch must have a matching mask filename, e.g. lake01_000.npy.
Split patches by lake/date BEFORE training; do not randomly split neighbouring patches
from the same scene into train and validation.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp

DATA = Path("data/patches")
IMAGES = DATA / "images"
MASKS = DATA / "masks"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class WaterPatchDataset(Dataset):
    def __init__(self):
        self.image_paths = sorted(IMAGES.glob("*.npy"))
        if not self.image_paths:
            raise RuntimeError(
                "No image patches found. Prepare data/patches/images/*.npy and "
                "data/patches/masks/*.npy first."
            )
        self.pairs = []
        for p in self.image_paths:
            m = MASKS / p.name
            if m.exists():
                self.pairs.append((p, m))
        if not self.pairs:
            raise RuntimeError("No matching masks found in data/patches/masks.")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        x = np.load(ip).astype("float32")
        y = np.load(mp).astype("float32")
        if x.ndim != 3:
            raise ValueError(f"{ip} must have shape (C,H,W), got {x.shape}")
        if y.ndim == 2:
            y = y[None, ...]
        # Input scaling should be consistent with the preprocessing used at inference.
        # If source data are Sentinel-2 integer reflectance scaled by 10000, divide by 10000
        # during patch preparation rather than applying arbitrary per-image normalization here.
        y = (y > 0).astype("float32")
        return torch.from_numpy(x), torch.from_numpy(y)

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WaterPatchDataset()
    if len(ds) < 5:
        raise RuntimeError("Collect more labelled patches before training/validation; at least 5 are required for this demo.")
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Change in_channels to match your input stack (e.g. 3 for RGB or 4 for B3/B4/B8/B11).
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        activation=None,
    ).to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val = float("inf")

    for epoch in range(1, 21):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice(logits, y)
                val_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d}/20 | train loss={tr:.4f} | val loss={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "in_channels": 4,
                 "epoch": epoch,
                 "val_loss": va},
                MODEL_DIR / "aquawatch_unet.pth"
            )
    print(f"Saved best checkpoint to {MODEL_DIR / 'aquawatch_unet.pth'}")

if __name__ == "__main__":
    main()
