import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.io import MemoryFile
except ImportError:
    rasterio = None

st.set_page_config(page_title="AquaWatch AI", page_icon="🌊", layout="wide")
st.title("🌊 AquaWatch AI")
st.caption("Satellite-based lake water-spread comparison — Bengaluru lakes and tanks")

st.warning(
    "A decrease in mapped water area is not proof of encroachment. "
    "Seasonality, rainfall, water levels, shadows, cloud and image alignment can affect results. "
    "Treat flagged zones as candidates for human review."
)

with st.sidebar:
    st.header("Comparison details")
    lake_name = st.text_input("Lake / tank name", "Example Bengaluru lake")
    old_year = st.number_input("Older image year", min_value=2000, max_value=2100, value=2024)
    new_year = st.number_input("Newer image year", min_value=2000, max_value=2100, value=2025)
    st.markdown("---")
    st.write("Best input: cloud-screened, georeferenced multispectral GeoTIFFs from comparable seasons.")
    st.write("For Bengaluru, start with the same lake and similar months in both years.")

left, right = st.columns(2)
with left:
    old_file = st.file_uploader("Upload older image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="old")
with right:
    new_file = st.file_uploader("Upload newer image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="new")

def read_image(upload):
    raw = upload.getvalue()
    name = upload.name.lower()
    if name.endswith((".tif", ".tiff")) and rasterio is not None:
        with MemoryFile(raw) as memfile:
            with memfile.open() as ds:
                arr = ds.read()
                profile = ds.profile.copy()
                descriptions = ds.descriptions
                crs = str(ds.crs) if ds.crs else None
                transform = ds.transform
        return {"kind": "raster", "array": arr, "profile": profile,
                "descriptions": descriptions, "crs": crs, "transform": transform}
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return {"kind": "rgb", "array": np.asarray(img), "image": img}

def rgb_preview(item):
    if item["kind"] == "rgb":
        return item["array"]
    arr = item["array"]
    # Display first 3 bands as a simple preview; not necessarily true colour.
    if arr.shape[0] >= 3:
        bands = arr[:3].astype(np.float32)
        out = np.moveaxis(bands, 0, -1)
        lo, hi = np.nanpercentile(out, [2, 98])
        if hi > lo:
            out = np.clip((out - lo) / (hi - lo), 0, 1)
        return (out * 255).astype(np.uint8)
    return arr[0]

def get_ndwi(item, green_band, nir_band):
    if item["kind"] != "raster":
        raise ValueError("NDWI needs a multispectral GeoTIFF, not a normal RGB image.")
    arr = item["array"].astype(np.float32)
    if arr.shape[0] < max(green_band, nir_band):
        raise ValueError(f"Image has {arr.shape[0]} bands; selected band is unavailable.")
    green = arr[green_band - 1]
    nir = arr[nir_band - 1]
    denom = green + nir
    ndwi = np.divide(green - nir, denom, out=np.zeros_like(green), where=np.abs(denom) > 1e-8)
    return ndwi

def valid_geo(item):
    return (item["kind"] == "raster" and item.get("crs") not in (None, "None")
            and item["profile"].get("transform") is not None)

if old_file and new_file:
    old = read_image(old_file)
    new = read_image(new_file)

    st.subheader(f"{lake_name}: {old_year} vs {new_year}")
    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb_preview(old), caption=f"Older image ({old_year})", use_container_width=True)
    with c2:
        st.image(rgb_preview(new), caption=f"Newer image ({new_year})", use_container_width=True)

    if old["kind"] == "raster" and new["kind"] == "raster":
        st.subheader("Water-mask settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            green_band = st.number_input("Green band index (1-based)", min_value=1, value=3, step=1)
        with col2:
            nir_band = st.number_input("NIR band index (1-based)", min_value=1, value=8, step=1)
        with col3:
            threshold = st.slider("NDWI threshold", min_value=-0.5, max_value=0.8, value=0.0, step=0.05)

        st.info(
            "Band indices depend on the GeoTIFF's band order. For Sentinel-2 band stacks, "
            "verify the order before using B3 (green) and B8 (NIR). If your file has only selected "
            "bands, enter their actual positions in that file."
        )
        if st.button("Compare water spread", type="primary"):
            try:
                old_ndwi = get_ndwi(old, int(green_band), int(nir_band))
                new_ndwi = get_ndwi(new, int(green_band), int(nir_band))
                if old_ndwi.shape != new_ndwi.shape:
                    st.error("The rasters have different dimensions. Reproject/resample both to the same grid before comparison.")
                elif old.get("crs") != new.get("crs") or old.get("transform") != new.get("transform"):
                    st.error("The rasters do not share the same CRS and pixel transform. Align them before comparing.")
                else:
                    old_valid = np.isfinite(old_ndwi)
                    new_valid = np.isfinite(new_ndwi)
                    valid = old_valid & new_valid
                    old_mask = (old_ndwi > threshold) & valid
                    new_mask = (new_ndwi > threshold) & valid
                    lost = old_mask & ~new_mask
                    gained = ~old_mask & new_mask
                    px_x = abs(float(old["transform"].a))
                    px_y = abs(float(old["transform"].e))
                    # Area calculation is meaningful only for projected CRS in metres.
                    crs_text = old.get("crs") or ""
                    projected_metres = ("EPSG:326" in crs_text or "EPSG:327" in crs_text or
                                        "EPSG:3857" in crs_text or "metre" in crs_text.lower() or
                                        "meter" in crs_text.lower())
                    if not projected_metres:
                        st.warning("CRS may not use metres. Reproject to an appropriate projected CRS before trusting area measurements.")
                    pixel_area = px_x * px_y
                    old_area = int(old_mask.sum()) * pixel_area
                    new_area = int(new_mask.sum()) * pixel_area
                    lost_area = int(lost.sum()) * pixel_area
                    gained_area = int(gained.sum()) * pixel_area
                    pct = ((old_area - new_area) / old_area * 100) if old_area else None

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Water area {old_year}", f"{old_area/10000:.2f} ha")
                    m2.metric(f"Water area {new_year}", f"{new_area/10000:.2f} ha")
                    m3.metric("Water spread lost", f"{lost_area/10000:.2f} ha")
                    m4.metric("Net area change", f"{pct:.1f}%" if pct is not None else "N/A")

                    st.write(f"Water spread gained: **{gained_area/10000:.2f} hectares**")
                    st.caption("Area values are estimates from classified pixels. They depend on correct bands, threshold, CRS, grid alignment, masks and image quality.")

                    # Colour map: stable water blue, lost red, gained green, other land transparent/dark.
                    change = np.zeros((*old_mask.shape, 3), dtype=np.uint8)
                    change[old_mask & new_mask] = [35, 119, 212]
                    change[lost] = [229, 72, 77]
                    change[gained] = [46, 157, 102]
                    st.image(change, caption="Blue = water in both years; red = water loss; green = water gain", use_container_width=True)

                    buf = io.BytesIO()
                    plt.figure(figsize=(8, 6))
                    plt.imshow(change)
                    plt.title(f"{lake_name}: water spread change {old_year}–{new_year}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(buf, format="png", dpi=160)
                    plt.close()
                    st.download_button("Download change map (PNG)", data=buf.getvalue(),
                                       file_name=f"aquawatch_change_{old_year}_{new_year}.png",
                                       mime="image/png")
                    st.warning("Red zones indicate mapped water spread loss, not confirmed illegal encroachment. Check comparable-season imagery, official lake boundaries and field evidence.")
            except Exception as exc:
                st.error(f"Could not process these files: {exc}")
    else:
        st.warning(
            "These are ordinary RGB images. This prototype can preview them, but it will not report real-world water area from them. "
            "For reliable NDWI and area measurement, upload aligned multispectral, georeferenced GeoTIFFs. "
            "An RGB segmentation model can be added after you create and validate a labelled training dataset."
        )
else:
    st.info("Upload both images to begin. For the first test, use two aligned Sentinel-2 multispectral GeoTIFFs from the same lake and similar seasons.")

st.markdown("---")
st.caption("AquaWatch AI • Prototype • Not a legal determination of encroachment")
