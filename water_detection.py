"""Water mask generation: U-Net inference with an index-based fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from unet_model import UNet, load_checkpoint
from water_index import clean_mask, index_to_mask, ndwi, ndwi_rgb

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "water_unet.pt"

METHODS = {
    "auto": "Auto (U-Net if a checkpoint exists, else water index)",
    "unet": "U-Net (deep learning)",
    "index": "Water index (threshold)",
    "ndwi": "NDWI (needs a NIR band)",
}


class WaterDetector:
    """Produces binary water masks for RGB (and optional NIR) scenes."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: UNet | None = None
        self.meta: dict[str, Any] = {}
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        if self.checkpoint_path and self.checkpoint_path.exists():
            self.model, self.meta = load_checkpoint(self.checkpoint_path, self.device)
            self.model.eval()

    @property
    def unet_available(self) -> bool:
        return self.model is not None

    @property
    def description(self) -> str:
        if not self.unet_available:
            return "Water index baseline (no U-Net checkpoint loaded)"
        iou = self.meta.get("val_iou")
        trained = self.meta.get("trained_on", "unknown data")
        iou_text = f", validation IoU {iou:.3f}" if isinstance(iou, (int, float)) else ""
        return f"U-Net checkpoint: {self.checkpoint_path.name} ({trained}{iou_text})"

    def predict_unet(self, rgb: np.ndarray, min_area_px: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Run the U-Net and return ``(mask, probability_map)``."""
        if self.model is None:
            raise RuntimeError("No U-Net checkpoint is loaded; train one with train_unet.py")

        image = np.asarray(rgb)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] != 3:
            image = image[..., :3]
        height, width = image.shape[:2]

        multiple = self.model.size_multiple
        pad_y = (-height) % multiple
        pad_x = (-width) % multiple
        padded = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")

        tensor = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32) / 255.0
        )[None].to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()[:height, :width]

        mask = clean_mask((probability >= 0.5).astype(np.uint8) * 255, min_area_px=min_area_px)
        return mask, probability

    def mask(
        self,
        rgb: np.ndarray,
        nir: np.ndarray | None = None,
        method: str = "auto",
        threshold: float | None = None,
        min_area_px: int = 32,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return ``(mask, info)`` for one scene."""
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")

        if method == "auto":
            method = "unet" if self.unet_available else "index"
        if method == "unet" and not self.unet_available:
            raise RuntimeError("U-Net requested but no checkpoint is loaded")
        if method == "ndwi" and nir is None:
            method = "index"

        info: dict[str, Any] = {"requested_method": method}

        if method == "unet":
            mask, probability = self.predict_unet(rgb, min_area_px=min_area_px)
            info.update(method="unet", threshold=0.5, water_fraction=float(mask.mean() / 255.0))
            info["probability"] = probability
            return mask, info

        if method == "ndwi":
            image = rgb.astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0
            green = image[..., 1]
            index = ndwi(green, nir)
            info["index_name"] = "NDWI (Green, NIR)"
        else:
            index = ndwi_rgb(rgb)
            info["index_name"] = "RGB water index (G-R)/(G+R)"

        mask, used_threshold = index_to_mask(index, threshold=threshold, min_area_px=min_area_px)
        info.update(
            method=method, threshold=used_threshold, water_fraction=float(mask.mean() / 255.0)
        )
        info["index"] = index
        return mask, info

    def mask_with_fallback(
        self,
        rgb: np.ndarray,
        nir: np.ndarray | None = None,
        method: str = "auto",
        threshold: float | None = None,
        min_area_px: int = 32,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Like :meth:`mask`, but silently falls back to the water index."""
        try:
            return self.mask(rgb, nir=nir, method=method, threshold=threshold, min_area_px=min_area_px)
        except RuntimeError:
            mask, info = self.mask(
                rgb, nir=nir, method="ndwi" if nir is not None else "index",
                threshold=threshold, min_area_px=min_area_px,
            )
            info["fallback"] = "U-Net unavailable, used the water index baseline"
            return mask, info


def resize_longest_side(rgb: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Downscale large uploads so analysis stays interactive."""
    height, width = rgb.shape[:2]
    if max(height, width) <= max_side:
        return rgb
    scale = max_side / max(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)