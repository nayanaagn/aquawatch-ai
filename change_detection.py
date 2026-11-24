"""Change detection and encroachment screening between two water masks.

Given a "before" and an "after" water mask (plus the after-date imagery), this
module builds:

* the lost-water, gained-water and stable-water layers;
* headline area statistics in hectares;
* a colour-coded change map for reporting;
* *encroachment candidates* - groups of lost-water pixels whose surface in the
  after image reads as vegetation, bare soil or built-up land rather than as
  damp shoreline. This is a screening heuristic, not a legal finding:
  candidates must be confirmed by field survey or revenue records.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from water_index import land_signal

# Colours are RGB so results render directly in Streamlit/PIL.
COLOURS = {
    "land": (0, 0, 0),
    "stable_water": (36, 96, 168),
    "lost_water": (214, 58, 58),
    "gained_water": (64, 196, 196),
    "candidate": (255, 214, 0),
}


def _as_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == np.uint8 and array.max(initial=0) > 1:
        return array > 127
    return array.astype(bool)


def analyse_change(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    after_rgb: np.ndarray | None = None,
    m2_per_px: float = 100.0,
    min_zone_px: int = 200,
    min_land_score: float = 0.30,
    max_candidates: int = 25,
) -> dict[str, Any]:
    """Compare two water masks and describe the change.

    Returns a dict with ``masks``, ``stats``, ``candidates``, ``change_map`` and
    ``overlay`` (the last two as uint8 RGB images).
    """
    before = _as_mask(before_mask)
    after = _as_mask(after_mask)
    if before.shape != after.shape:
        raise ValueError(f"Mask shapes differ: {before.shape} vs {after.shape}")

    lost = before & ~after
    gained = after & ~before
    stable = before & after

    hectares = m2_per_px / 10_000.0
    before_px = int(before.sum())
    after_px = int(after.sum())
    lost_px = int(lost.sum())
    gained_px = int(gained.sum())

    stats: dict[str, Any] = {
        "water_before_ha": before_px * hectares,
        "water_after_ha": after_px * hectares,
        "lost_ha": lost_px * hectares,
        "gained_ha": gained_px * hectares,
        "net_change_ha": (after_px - before_px) * hectares,
        "percent_change": ((after_px - before_px) / before_px * 100.0) if before_px else 0.0,
        "water_before_px": before_px,
        "water_after_px": after_px,
        "lost_px": lost_px,
        "gained_px": gained_px,
        "m2_per_px": m2_per_px,
        "image_px": int(before.size),
    }

    candidates = _encroachment_candidates(
        lost=lost,
        after_rgb=after_rgb,
        hectares=hectares,
        min_zone_px=min_zone_px,
        min_land_score=min_land_score,
        max_candidates=max_candidates,
    )
    stats["candidate_count"] = len(candidates)
    stats["candidate_area_ha"] = float(sum(c["area_ha"] for c in candidates))

    change_map = np.zeros((*before.shape, 3), dtype=np.uint8)
    change_map[stable] = COLOURS["stable_water"]
    change_map[gained] = COLOURS["gained_water"]
    change_map[lost] = COLOURS["lost_water"]

    overlay = None
    if after_rgb is not None:
        overlay = _overlay(after_rgb, stable, gained, lost, candidates, before.shape)

    return {
        "masks": {"before": before, "after": after, "lost": lost, "gained": gained, "stable": stable},
        "stats": stats,
        "candidates": candidates,
        "change_map": change_map,
        "overlay": overlay,
    }


def _encroachment_candidates(
    lost: np.ndarray,
    after_rgb: np.ndarray | None,
    hectares: float,
    min_zone_px: int,
    min_land_score: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Group lost-water pixels that now look like land into candidate parcels.

    Seasonal retreat exposes damp lake bed, which has a low land signal; fill,
    construction and vegetation score high. Only the high-signal pixels are
    grouped, so a shrinking lake with one filled parcel yields one candidate
    rather than one giant blob.
    """
    if after_rgb is None or not lost.any():
        return []

    signal = land_signal(after_rgb)
    if signal.shape[:2] != lost.shape:
        signal = cv2.resize(signal, (lost.shape[1], lost.shape[0]), interpolation=cv2.INTER_AREA)

    converted = (lost & (signal >= float(min_land_score))).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    converted = cv2.morphologyEx(converted, cv2.MORPH_OPEN, kernel)
    converted = cv2.morphologyEx(converted, cv2.MORPH_CLOSE, kernel)

    count, labels, component_stats, centroids = cv2.connectedComponentsWithStats(converted, connectivity=8)

    found: list[dict[str, Any]] = []
    for label in range(1, count):
        area_px = int(component_stats[label, cv2.CC_STAT_AREA])
        if area_px < min_zone_px:
            continue

        zone = labels == label
        contours, _ = cv2.findContours(
            zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
        solidity = min(float(area_px / hull_area), 1.0) if hull_area > 0 else 0.0

        cx, cy = centroids[label]
        found.append(
            {
                "zone": len(found) + 1,
                "area_ha": area_px * hectares,
                "area_px": area_px,
                "centroid_x": int(round(cx)),
                "centroid_y": int(round(cy)),
                "solidity": round(solidity, 3),
                "land_score": round(float(signal[zone].mean()), 3),
                "bbox": (
                    int(component_stats[label, cv2.CC_STAT_LEFT]),
                    int(component_stats[label, cv2.CC_STAT_TOP]),
                    int(component_stats[label, cv2.CC_STAT_WIDTH]),
                    int(component_stats[label, cv2.CC_STAT_HEIGHT]),
                ),
                "mask": zone,
            }
        )

    found.sort(key=lambda item: item["area_ha"], reverse=True)
    for index, candidate in enumerate(found[:max_candidates], start=1):
        candidate["zone"] = index
    return found[:max_candidates]


def _overlay(
    after_rgb: np.ndarray,
    stable: np.ndarray,
    gained: np.ndarray,
    lost: np.ndarray,
    candidates: list[dict[str, Any]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Dim the after-date image and tint the change layers on top of it."""
    base = np.asarray(after_rgb, dtype=np.float32)
    if base.shape[:2] != shape:
        base = cv2.resize(base, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    overlay = base * 0.45

    for layer, colour, alpha in (
        (stable, COLOURS["stable_water"], 0.25),
        (gained, COLOURS["gained_water"], 0.75),
        (lost, COLOURS["lost_water"], 0.60),
    ):
        colour_array = np.array(colour, dtype=np.float32)
        overlay[layer] = overlay[layer] * (1 - alpha) + colour_array * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    for candidate in candidates:
        contours, _ = cv2.findContours(
            candidate["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, COLOURS["candidate"], 2)
        x, y, _, _ = candidate["bbox"]
        cv2.putText(
            overlay,
            f"Z{candidate['zone']}",
            (x + 2, max(12, y + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOURS["candidate"],
            1,
            cv2.LINE_AA,
        )
    return overlay