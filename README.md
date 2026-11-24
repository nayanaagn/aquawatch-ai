# AquaWatch AI — Starter Project

A beginner-friendly prototype for comparing water spread in two lake/tank satellite images.

## What works in this starter

- Uploads two files.
- Previews RGB images and GeoTIFF rasters.
- For matching, georeferenced multispectral GeoTIFFs, calculates NDWI masks.
- Compares water loss and gain and estimates pixel area.
- Exports a PNG change map.
- Includes a starter U-Net training script and dataset layout.

## Important limitations

- This is a prototype, not a validated encroachment detector.
- Red pixels mean mapped water spread loss, not proof of illegal construction or land encroachment.
- NDWI band positions depend on how your GeoTIFF was exported. Confirm band order.
- Images must be aligned and use a projected coordinate reference system in metres for trustworthy area calculation.
- The app currently uses NDWI as its working baseline. It does not automatically use the trained U-Net checkpoint yet.
- RGB screenshots do not have the spectral information required for NDWI area analysis.
- The included training script expects prepared `.npy` patches; patch extraction and expert/verified mask creation are separate preprocessing tasks.

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Recommended first data format

Use Sentinel-2 surface-reflectance GeoTIFFs containing known bands, ideally B3 (green), B4 (red), B8 (NIR), and B11 (SWIR). Export both years with the same band order, projection, resolution, extent and grid. For NDWI, use B3 and B8. For MNDWI, use B3 and B11, after accounting for B11's native 20 m resolution.

Use comparable seasons and cloud-screened data. Mask cloud/shadow/no-data pixels before calculating area. Do not compare images until their CRS, transform, dimensions and pixel alignment are confirmed.

## Training data layout

```text
data/patches/
├── images/
│   ├── lake01_000.npy  # shape (4, 256, 256), reflectance consistently scaled
│   └── lake02_000.npy
└── masks/
    ├── lake01_000.npy  # shape (256, 256), 1=water, 0=non-water
    └── lake02_000.npy
```

Masks should be reviewed reference labels, not blindly copied from NDWI outputs. Include water, wet soil, shadows, vegetation, buildings and different seasons. Split data by lake or scene before training to avoid leakage.

Run training after preparing data:

```bash
python train_unet.py
```

The current training script uses four input channels. If your patch channels differ, update `in_channels` and the saved checkpoint metadata. The simple random split is only a code demonstration; for real evaluation, split by lake/scene before training.

## Suggested next milestones

1. Test the app with two correctly aligned GeoTIFFs from the same lake.
2. Validate the NDWI threshold against manually reviewed masks.
3. Add cloud/no-data masking and a robust reprojection/alignment pipeline.
4. Prepare reviewed labelled patches and train U-Net.
5. Add U-Net inference to the app using exactly the same band scaling and preprocessing as training.
6. Add official lake boundary overlays and a human-review status for suspected encroachment.
7. Evaluate on lakes/dates not used for training before presenting accuracy claims.
