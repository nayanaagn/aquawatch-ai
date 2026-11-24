"""AquaWatch AI - satellite water body monitoring dashboard.

Detects water bodies with a U-Net (or a classical water index), compares two
acquisition dates and screens lost-water zones for encroachment.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import synthetic
from change_detection import analyse_change
from water_detection import DEFAULT_CHECKPOINT, METHODS, WaterDetector, resize_longest_side
from water_index import area_hectares, stretch_percentiles

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"
IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "webp"]

st.set_page_config(page_title="AquaWatch AI", page_icon="🛰️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_detector(checkpoint_mtime: float | None) -> WaterDetector:
    """Load the U-Net once per session (``checkpoint_mtime`` busts the cache)."""
    return WaterDetector(DEFAULT_CHECKPOINT if checkpoint_mtime else None)


@st.cache_data(show_spinner=False)
def demo_scene(name: str) -> np.ndarray:
    path = SAMPLE_DIR / name
    if not path.exists():
        synthetic.write_demo_scenes(SAMPLE_DIR)
    return np.asarray(Image.open(path).convert("RGB")).copy()


def checkpoint_mtime() -> float | None:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT.stat().st_mtime
    return None


# --------------------------------------------------------------------------- #
# Small image helpers
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image).copy()


def decode_band(data: bytes) -> np.ndarray:
    band = Image.open(io.BytesIO(data))
    if band.mode == "P":
        band = band.convert("RGB")
    return np.asarray(band.convert("L")).copy()


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    """Render a water mask as blue-on-dark for display."""
    water = np.asarray(mask) > 127
    out = np.zeros((*water.shape, 3), dtype=np.uint8)
    out[...] = (18, 22, 28)
    out[water] = (46, 132, 214)
    return out


def match_shapes(after: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if after.shape[:2] == shape:
        return after
    return cv2.resize(after, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if k != "mask"}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar(detector: WaterDetector) -> dict[str, Any]:
    st.sidebar.header("Detection")
    method = st.sidebar.selectbox(
        "Water detection method",
        list(METHODS),
        format_func=lambda key: METHODS[key],
        index=0,
    )
    auto_threshold = st.sidebar.checkbox("Automatic threshold (Otsu)", value=True)
    threshold = st.sidebar.slider(
        "Water index threshold",
        min_value=-0.40,
        max_value=0.80,
        value=0.05,
        step=0.01,
        disabled=auto_threshold,
        help="Only used for the index methods; ignored by the U-Net.",
    )
    min_area = int(
        st.sidebar.number_input("Minimum water body (pixels)", 16, 200_000, 200, step=50)
    )
    harmonise = st.sidebar.checkbox(
        "Harmonise brightness between dates",
        value=True,
        help="Percentile stretch per date, reduces false change from exposure/atmosphere.",
    )

    st.sidebar.header("Change analysis")
    m2_per_px = float(
        st.sidebar.number_input(
            "Pixel area (m²)", 1.0, 10_000.0, 100.0, step=10.0,
            help="Sentinel-2 at 10 m resolution = 100 m².",
        )
    )
    min_zone = int(
        st.sidebar.number_input("Minimum change zone (pixels)", 16, 200_000, 400, step=50)
    )
    min_land_score = float(
        st.sidebar.slider(
            "Encroachment land-signal threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Lost-water zones whose surface reads at least this much like land "
            "(vegetation, soil, built-up) are flagged as candidates.",
        )
    )

    st.sidebar.divider()
    st.sidebar.caption(detector.description)
    if not detector.unet_available:
        st.sidebar.warning(
            "No U-Net checkpoint found. Run `python train_unet.py` to create "
            "`models/water_unet.pt`, or keep using the water index."
        )

    return {
        "method": method,
        "threshold": None if auto_threshold else threshold,
        "min_area": min_area,
        "harmonise": harmonise,
        "m2_per_px": m2_per_px,
        "min_zone": min_zone,
        "min_land_score": min_land_score,
    }


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def change_view(detector: WaterDetector, settings: dict[str, Any]) -> None:
    left, right = st.columns(2)
    before_file = left.file_uploader("Before image (earlier date)", IMAGE_TYPES, key="before_file")
    after_file = right.file_uploader("After image (later date)", IMAGE_TYPES, key="after_file")
    nir_file = st.file_uploader(
        "Optional NIR band (enables true NDWI; must cover the same area)",
        IMAGE_TYPES,
        key="nir_file",
    )
    use_demo = st.checkbox(
        "Use the bundled demo scenes (same lake, two dates)",
        value=not (before_file or after_file),
        disabled=bool(before_file or after_file),
    )

    if before_file and after_file:
        before_rgb = resize_longest_side(decode_image(before_file.getvalue()))
        after_rgb = resize_longest_side(decode_image(after_file.getvalue()))
        source = "your uploads"
    elif use_demo:
        before_rgb = demo_scene("before.png")
        after_rgb = demo_scene("after.png")
        source = "bundled demo scenes"
    else:
        st.info("Upload a before and an after image, or tick the demo scenes box.")
        return

    after_rgb = match_shapes(after_rgb, before_rgb.shape[:2])
    before_nir = after_nir = None
    if nir_file is not None:
        nir = resize_longest_side(decode_band(nir_file.getvalue()))
        nir = cv2.resize(nir, (before_rgb.shape[1], before_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        before_nir = after_nir = nir

    overlay_source = before_rgb
    before_proc = stretch_percentiles(before_rgb) if settings["harmonise"] else before_rgb
    after_proc = stretch_percentiles(after_rgb) if settings["harmonise"] else after_rgb

    if settings["harmonise"]:
        overlay_source = after_proc

    with st.spinner("Segmenting water and comparing dates..."):
        before_mask, before_info = detector.mask_with_fallback(
            before_proc,
            nir=before_nir,
            method=settings["method"],
            threshold=settings["threshold"],
            min_area_px=settings["min_area"],
        )
        after_mask, after_info = detector.mask_with_fallback(
            after_proc,
            nir=after_nir,
            method=settings["method"],
            threshold=settings["threshold"],
            min_area_px=settings["min_area"],
        )
        result = analyse_change(
            before_mask,
            after_mask,
            after_rgb=overlay_source,
            m2_per_px=settings["m2_per_px"],
            min_zone_px=settings["min_zone"],
            min_land_score=settings["min_land_score"],
        )

    stats = result["stats"]
    for info in (before_info, after_info):
        if info.get("fallback"):
            st.warning(info["fallback"])

    st.caption(
        f"Source: {source} · {before_rgb.shape[1]}×{before_rgb.shape[0]} px · "
        f"method: {before_info['method']} ({before_info.get('index_name', 'U-Net')}) · "
        f"threshold: {before_info['threshold']:.3f}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Water before", f"{stats['water_before_ha']:.0f} ha")
    m2.metric("Water after", f"{stats['water_after_ha']:.0f} ha")
    m3.metric(
        "Net change",
        f"{stats['net_change_ha']:+.0f} ha",
        delta=f"{stats['percent_change']:+.1f} %",
        delta_color="inverse",
    )
    zones = int(stats["candidate_count"])
    m4.metric(
        "Candidates",
        f"{zones} zone{'s' if zones != 1 else ''}",
        delta=f"{stats['candidate_area_ha']:.0f} ha",
        delta_color="inverse",
    )

    st.subheader("Water detection")
    col_b, col_bm, col_a, col_am = st.columns(4)
    col_b.image(before_rgb, caption="Before", use_container_width=True)
    col_bm.image(mask_rgb(before_mask), caption="Water mask (before)", use_container_width=True)
    col_a.image(after_rgb, caption="After", use_container_width=True)
    col_am.image(mask_rgb(after_mask), caption="Water mask (after)", use_container_width=True)

    st.subheader("Change detection")
    col_map, col_overlay = st.columns(2)
    col_map.image(result["change_map"], caption="Change map (red: lost, cyan: gained, blue: stable water)", use_container_width=True)
    if result["overlay"] is not None:
        col_overlay.image(
            result["overlay"],
            caption="Overlay on the after image (yellow: encroachment candidates)",
            use_container_width=True,
        )

    st.subheader("Encroachment candidates")
    if result["candidates"]:
        rows = [
            {
                "Zone": c["zone"],
                "Area (ha)": round(c["area_ha"], 3),
                "Land signal": c["land_score"],
                "Solidity": c["solidity"],
                "Centre (x, y)": f"{c['centroid_x']}, {c['centroid_y']}",
            }
            for c in result["candidates"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Candidates are groups of lost water whose surface in the later image reads "
            "like vegetation, bare soil or built-up land, so damp seasonal retreat is not "
            "listed. This is screening evidence only - confirm on the ground before acting."
        )
    else:
        st.info(
            "No lost-water area above the minimum size reads as land in the later image. "
            "Lower the land-signal threshold, or check the change map for retreat."
        )

    with st.expander("Downloads"):
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Change map (PNG)", png_bytes(result["change_map"]), "change_map.png", "image/png")
        if result["overlay"] is not None:
            d2.download_button("Overlay (PNG)", png_bytes(result["overlay"]), "change_overlay.png", "image/png")
        d3.download_button("Before mask (PNG)", png_bytes(before_mask), "before_mask.png", "image/png")
        d4.download_button("After mask (PNG)", png_bytes(after_mask), "after_mask.png", "image/png")

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["zone", "area_ha", "area_px", "land_score", "solidity", "centroid_x", "centroid_y"])
        for c in result["candidates"]:
            writer.writerow([
                c["zone"], f"{c['area_ha']:.4f}", c["area_px"], c["land_score"], c["solidity"],
                c["centroid_x"], c["centroid_y"],
            ])
        st.download_button("Zones (CSV)", csv_buffer.getvalue(), "lost_water_zones.csv", "text/csv")
        report = {
            "source": source,
            "settings": {k: v for k, v in settings.items()},
            "before": {k: v for k, v in before_info.items() if k in ("method", "index_name", "threshold", "water_fraction")},
            "after": {k: v for k, v in after_info.items() if k in ("method", "index_name", "threshold", "water_fraction")},
            "stats": json_safe(stats),
            "candidates": json_safe(result["candidates"]),
        }
        st.download_button("Report (JSON)", json.dumps(report, indent=2), "aquawatch_report.json", "application/json")


def single_view(detector: WaterDetector, settings: dict[str, Any]) -> None:
    uploaded = st.file_uploader("Satellite image", IMAGE_TYPES, key="single_file")
    if uploaded is not None:
        rgb = resize_longest_side(decode_image(uploaded.getvalue()))
    else:
        rgb = demo_scene("after.png")
        st.caption("Showing the bundled demo scene; upload an image to analyse your own.")

    processed = stretch_percentiles(rgb) if settings["harmonise"] else rgb
    mask, info = detector.mask_with_fallback(
        processed,
        method=settings["method"],
        threshold=settings["threshold"],
        min_area_px=settings["min_area"],
    )

    col_a, col_b = st.columns(2)
    col_a.image(rgb, caption="Input", use_container_width=True)
    col_b.image(mask_rgb(mask), caption="Detected water", use_container_width=True)
    st.metric("Water area", f"{area_hectares(mask, settings['m2_per_px']):.1f} ha")
    st.caption(f"Method: {info['method']} · water covers {mask.mean() / 255.0 * 100:.1f}% of the scene")
    st.download_button("Water mask (PNG)", png_bytes(mask), "water_mask.png", "image/png")


def about_view(detector: WaterDetector) -> None:
    st.markdown(
        """
### How the analysis works

1. **Water detection** - the U-Net segments water in each date. Without a trained
   checkpoint the app falls back to a water index (true NDWI when a NIR band is
   supplied, otherwise an RGB approximation). Masks are cleaned with
   morphological closing and small-blob removal.
2. **Change detection** - water masks are compared pixel by pixel into
   *lost*, *gained* and *stable* layers, with areas converted to hectares through
   the pixel area you set.
3. **Encroachment screening** - lost-water pixels whose surface in the later date
   looks land-like (excess-green and brightness) are grouped into parcels above
   the minimum size and outlined as *candidates*. Damp exposed lake bed scores
   low and is not listed, which keeps seasonal retreat out of the candidate list.

### What this does and does not tell you

* It reports **where water disappeared and what the surface looks like now**.
  Seasonal drawdown, drought, siltation and earth-filling all reduce water area;
  only expert review distinguishes them.
* Candidate zones are a prioritisation aid, not proof of illegal occupation.
  Confirm with high-resolution imagery, field survey or revenue records.
* The bundled checkpoint is trained on **synthetic scenes**, so it transfers
  imperfectly to real imagery. Retrain on labelled real chips
  (`train_unet.py` expects a dataset module with the same interface) before
  relying on it for decisions.
* Areas are only as accurate as the pixel-area setting; a wrong ground sampling
  distance scales every hectare figure.

### Real imagery sources

Fetch cloud-free Sentinel-2 L2A crops (10 m bands B02/B03/B04/B08) from the
Copernicus Data Space Browser or Google Earth Engine, then export PNG/GeoTIFF
crops for the same footprint on two dates. Uploading a NIR crop alongside the
RGB image enables true NDWI.
        """
    )
    st.caption(detector.description)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🛰️ AquaWatch AI")
    st.caption(
        "AI + satellite imagery for water body monitoring: detection, change "
        "detection and encroachment screening."
    )

    detector = get_detector(checkpoint_mtime())
    settings = sidebar(detector)

    tab_change, tab_single, tab_about = st.tabs(["Change analysis", "Single scene", "About"])
    with tab_change:
        change_view(detector, settings)
    with tab_single:
        single_view(detector, settings)
    with tab_about:
        about_view(detector)


if __name__ == "__main__":
    main()