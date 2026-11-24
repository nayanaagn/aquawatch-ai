"""Procedural satellite-like scenes with water masks.

These scenes stand in for real Sentinel-2 crops so the U-Net can be trained and
the app demoed without downloading imagery. They are *not* a substitute for
real training data (see the limitations section of README.md).

A scene can be rendered with a shrunken lake (``shrink``) and with a
straight-edged land parcel eating into the lake (``intrusion``). The exposed
lake bed between the old and new shoreline is drawn as damp mud, while the
intrusion is drawn as dry compacted earth with vegetation - that difference is
what lets the encroachment screening separate seasonal retreat from fill.

Run ``python synthetic.py --out sample_data --size 384`` to write the two demo
scenes used by the Streamlit app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _smooth_noise(
    size: int, rng: np.random.Generator, scales: tuple[int, ...] = (4, 12, 32)
) -> np.ndarray:
    """Multi-scale value noise in [0, 1]."""
    field = np.zeros((size, size), dtype=np.float32)
    weight = 0.0
    for i, grid in enumerate(scales):
        coarse = rng.random((grid, grid)).astype(np.float32)
        field += cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC) * (1.0 / (i + 1))
        weight += 1.0 / (i + 1)
    field /= weight
    return np.clip(field, 0.0, 1.0)


def _lake_fields(
    size: int, rng: np.random.Generator, shrink: float = 0.0
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Return ``(full_lake, current_lake, geometry)``.

    ``full_lake`` is the lake at its high-water extent, ``current_lake`` the
    lake after ``shrink`` (0 = no retreat) and ``geometry`` is
    ``(center_y, center_x, radius)``.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = float(rng.uniform(0.40, 0.60) * size)
    cx = float(rng.uniform(0.40, 0.60) * size)
    base_radius = float(rng.uniform(0.16, 0.30) * size)

    theta = np.arctan2(yy - cy, xx - cx)
    wobble = np.zeros_like(theta)
    for k in range(3):
        wobble += rng.uniform(-0.16, 0.16) * np.sin((k + 2) * theta + rng.uniform(0, 2 * np.pi))

    radius = base_radius * (1.0 + wobble)
    dist = np.hypot(yy - cy, xx - cx)

    full = dist <= radius
    current = dist <= radius * (1.0 - float(np.clip(shrink, 0.0, 0.85)))
    return full, current, (cy, cx, base_radius)


def _intrusion_patch(
    size: int, rng: np.random.Generator, geometry: tuple[float, float, float]
) -> np.ndarray:
    """A straight-edged land parcel that eats into the lake (an encroachment)."""
    cy, cx, radius = geometry
    angle = rng.uniform(0, 2 * np.pi)
    reach = radius * rng.uniform(0.55, 0.95)
    py = cy + reach * np.sin(angle)
    px = cx + reach * np.cos(angle)
    half_w = radius * rng.uniform(0.22, 0.40)
    half_h = radius * rng.uniform(0.18, 0.32)

    patch = np.zeros((size, size), dtype=bool)
    y0 = int(max(0, py - half_h))
    y1 = int(min(size, py + half_h))
    x0 = int(max(0, px - half_w))
    x1 = int(min(size, px + half_w))
    patch[y0:y1, x0:x1] = True
    return patch


def _land_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    vegetation = _smooth_noise(size, rng, scales=(8, 20, 48))
    dryness = _smooth_noise(size, rng, scales=(4, 16, 40))
    urban = _smooth_noise(size, rng, scales=(2, 6, 18))

    veg_colour = np.array([74.0, 108.0, 58.0], dtype=np.float32)
    dry_colour = np.array([172.0, 152.0, 116.0], dtype=np.float32)
    urban_colour = np.array([148.0, 146.0, 140.0], dtype=np.float32)

    mix_veg = np.clip(vegetation * 1.3 - 0.25, 0, 1)[..., None]
    base = dry_colour * (1 - mix_veg) + veg_colour * mix_veg

    urban_mask = np.clip((urban - 0.72) * 4.0, 0, 1)[..., None]
    base = base * (1 - urban_mask * 0.7) + urban_colour * urban_mask * 0.7

    tone = (0.82 + 0.36 * dryness)[..., None]
    return np.clip(base * tone, 0, 255)


def _water_texture(size: int, rng: np.random.Generator, lake: np.ndarray) -> np.ndarray:
    deep = np.array([28.0, 62.0, 88.0], dtype=np.float32)
    shallow = np.array([54.0, 96.0, 108.0], dtype=np.float32)

    depth = cv2.distanceTransform(lake.astype(np.uint8), cv2.DIST_L2, 5)
    depth /= max(float(depth.max()), 1e-6)
    depth = np.clip(depth * 2.2, 0, 1)[..., None]

    ripple = _smooth_noise(size, rng, scales=(5, 15, 45))[..., None]
    water = shallow * (1 - depth) + deep * depth
    water = water * (0.90 + 0.20 * ripple)
    # Turbid shallow edges read slightly greener.
    water[..., 1] += 14.0 * (1.0 - depth[..., 0])
    return np.clip(water, 0, 255)


def _mud_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Damp exposed lake bed: dark, low vegetation signal."""
    dampness = _smooth_noise(size, rng, scales=(6, 16, 40))[..., None]
    mud = np.array([104.0, 92.0, 76.0], dtype=np.float32)
    silt = np.array([132.0, 120.0, 98.0], dtype=np.float32)
    base = mud * (1.0 - dampness * 0.5) + silt * (dampness * 0.5)
    return np.clip(base * (0.85 + 0.22 * dampness), 0, 255)


def _fill_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Compacted earth with vegetation patches: a strong land signal."""
    mixture = _smooth_noise(size, rng, scales=(3, 9, 22))[..., None]
    soil = np.array([178.0, 166.0, 142.0], dtype=np.float32)
    vegetation = np.array([78.0, 112.0, 60.0], dtype=np.float32)
    base = soil * (1.0 - mixture * 0.35) + vegetation * (mixture * 0.35)
    grain = rng.normal(0.0, 4.0, (size, size, 1)).astype(np.float32)
    return np.clip(base * (0.95 + 0.10 * mixture) + grain, 0, 255)


def make_scene(
    seed: int = 0,
    size: int = 128,
    shrink: float = 0.0,
    intrusion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a scene.

    Returns ``(image, mask)`` where ``image`` is an ``H x W x 3`` uint8 RGB
    array and ``mask`` is uint8 with water at 255 and land at 0. Calling this
    twice with the same ``seed``/``size`` keeps the lake geometry and texture
    fixed, so the pair can be used as a before/after.
    """
    shape_rng = np.random.default_rng(1000 + int(seed))
    tex_rng = np.random.default_rng(2000 + int(seed))

    full, current, geometry = _lake_fields(size, shape_rng, shrink=shrink)
    fill = _intrusion_patch(size, shape_rng, geometry) if intrusion else np.zeros_like(current)

    lake = current & ~fill
    exposed = full & ~current & ~fill

    image = _land_texture(size, tex_rng)
    if exposed.any():
        image[exposed] = _mud_texture(size, tex_rng)[exposed]
    if intrusion and fill.any():
        image[fill] = _fill_texture(size, tex_rng)[fill]
    if lake.any():
        image[lake] = _water_texture(size, tex_rng, lake)[lake]

    image = np.clip(image + tex_rng.normal(0.0, 3.5, image.shape), 0, 255).astype(np.uint8)
    mask = lake.astype(np.uint8) * 255
    return image, mask


def write_demo_scenes(out_dir: str | Path, seed: int = 7, size: int = 384) -> tuple[Path, Path]:
    """Write the bundled before/after demo scenes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before, _ = make_scene(seed=seed, size=size, shrink=0.0)
    after, _ = make_scene(seed=seed, size=size, shrink=0.35, intrusion=True)
    before_path = out_dir / "before.png"
    after_path = out_dir / "after.png"
    cv2.imwrite(str(before_path), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(after_path), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    return before_path, after_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the demo before/after scenes.")
    parser.add_argument("--out", default="sample_data")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    paths = write_demo_scenes(args.out, seed=args.seed, size=args.size)
    print("wrote", ", ".join(str(p) for p in paths))