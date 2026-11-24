"""Classical water-index helpers and mask post-processing.

Two indices are provided:

* ``ndwi(green, nir)`` - the real Normalised Difference Water Index,
  ``(Green - NIR) / (Green + NIR)``, which needs a near-infrared band.
* ``ndwi_rgb(rgb)`` - an RGB approximation, ``(Green - Red) / (Green + Red)``,
  used when no NIR band is available. Water absorbs red and reflects green, so
  the index still separates most water from vegetation and soil, but it is a
  proxy, not a measurement.

Both return float arrays in roughly ``[-1, 1]`` where higher means wetter.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def to_float_rgb(image: np.ndarray) -> np.ndarray:
    """uint8 RGB (or BGR-free array) -> float32 [0, 1]."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """True NDWI from green and near-infrared bands."""
    g = np.asarray(green, dtype=np.float32)
    n = np.asarray(nir, dtype=np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    if n.max() > 1.0:
        n = n / 255.0
    return (g - n) / (g + n + EPS)


def ndwi_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB-only approximation of NDWI: ``(G - R) / (G + R)``."""
    x = to_float_rgb(rgb)
    red, green = x[..., 0], x[..., 1]
    return (green - red) / (green + red + EPS)


def land_signal(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel 0..1 evidence that a pixel is *not* water.

    Combines an excess-green (vegetation) term with a brightness term that
    responds to bare soil and built surfaces. Used to screen lost-water zones.
    """
    x = to_float_rgb(rgb)
    red, green, blue = x[..., 0], x[..., 1], x[..., 2]
    # Excess green is a weak signal on RGB alone; 1.6 lifts healthy vegetation
    # well clear of damp mud without pushing bare soil above the shoreline.
    vegetation = np.clip(1.6 * (2.0 * green - red - blue), 0.0, 1.0)
    built = np.clip((x.mean(axis=2) - 0.42) / 0.35, 0.0, 1.0)
    return np.maximum(vegetation, built)


def stretch_percentiles(rgb: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-channel percentile stretch, used to harmonise two acquisition dates."""
    x = np.asarray(rgb, dtype=np.float32)
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        band = x[..., c]
        lo, hi = np.percentile(band, (low, high))
        if hi - lo < EPS:
            out[..., c] = band
        else:
            out[..., c] = (band - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, min_area_px: int = 32, close_ksize: int = 3) -> np.ndarray:
    """Remove salt noise, close pinholes and drop tiny blobs."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros(binary.shape, dtype=np.uint8)

    if close_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if min_area_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
                keep[labels == label] = 1
        binary = keep

    return (binary * 255).astype(np.uint8)


def index_to_mask(
    index: np.ndarray,
    threshold: float | None = None,
    min_area_px: int = 32,
) -> tuple[np.ndarray, float]:
    """Threshold a water index into a clean water mask.

    ``threshold=None`` uses Otsu's method on the index image. Returns the mask
    (uint8, water = 255) and the threshold that was applied.
    """
    index = np.asarray(index, dtype=np.float32)
    scaled = np.clip((index + 1.0) * 0.5, 0.0, 1.0)
    octet = (scaled * 255.0).astype(np.uint8)

    if threshold is None:
        raw, _ = cv2.threshold(octet, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        level = float(raw)
    else:
        level = float(np.clip((threshold + 1.0) * 0.5, 0.0, 1.0) * 255.0)

    mask = (octet > level).astype(np.uint8) * 255
    return clean_mask(mask, min_area_px=min_area_px), level / 127.5 - 1.0


def area_hectares(mask: np.ndarray, m2_per_px: float) -> float:
    """Water area of a mask in hectares."""
    return float(np.count_nonzero(np.asarray(mask) > 0) * m2_per_px / 10_000.0)