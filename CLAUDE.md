# CLAUDE.md

This file provides guidance for Claude Code when working with the sat-tile-stack repository.

## Project Overview

sat-tile-stack is a Python package for constructing deep learning-ready time-series datasets from Sentinel-2 satellite imagery via Microsoft Planetary Computer's STAC catalog.

## Core Functionality

The main function `sattile_stack()` in `sattilestack/stack.py`:
- Searches the Planetary Computer STAC catalog for Sentinel-2 L2A scenes
- Builds a 4D xarray DataArray (time, band, y, x) using stackstac
- Resamples to daily cadence with cloud cover metadata
- Optionally applies IQR-based normalization
- Crops to a configurable tile size around the centroid
- Optionally adds cloud masks and spatial polygon masks

### Key Parameters
- `centroid`: (longitude, latitude) in decimal degrees
- `band_names`: Sentinel-2 bands (e.g., ['B04', 'B03', 'B02', 'B08', 'B11'])
- `pix_res`: pixel resolution in meters (default: 10)
- `tile_size`: output tile size in pixels (default: 1024)
- `time_range`: ISO8601 date range (default: '2019-05-01/2019-09-30')
- `cloudmask`: if True, adds pixel-wise cloud mask using Williamson2018b method
- `mask`: GeoDataFrame for spatial masking (e.g., lake polygons)

## Module Structure

```
sattilestack/
    __init__.py     # Package exports
    stack.py        # Main sattile_stack() function
    bounds.py       # Bounding box, CRS selection, and mask utilities
    utils.py        # Cloud masking and normalization functions
    visualize.py    # Movie/animation generation from timestacks
    io.py           # NetCDF writing utilities
```

## Key Technical Details

### CRS Selection (`bounds.py:best_crs_for_point`)
- |lat| >= 60: Polar Stereographic (EPSG:3413 North, EPSG:3031 South)
- Otherwise: UTM zone based on longitude

### Cloud Masking (`utils.py:cloud_pix_mask`)
- Williamson2018b method: SWIR1 (B11) > 0.140 indicates clouds
- Also marks NaN pixels as cloudy

### Normalization (`utils.py:combo_scaler`)
- Robust IQR-based normalization: (x - median) / IQR
- Then scaled to [0, range_max]

## Satellite Product Reference

The `pix_res` parameter controls output resolution. Setting it below native resolution
interpolates (no new information); above native downsamples.

**Sentinel-2 L2A** (`sentinel-2-l2a`): 10m bands (B02/B03/B04/B08), 20m bands (B05/B06/B07/B8A/B11/B12/SCL), 60m (B09). Reflectance scaled by 10000.
**Sentinel-1 GRD IW** (`sentinel-1-grd`): 10m pixel spacing but ~20m true resolution (oversampled). VV/VH or HH/HV. Backscatter in dB. No `eo:cloud_cover` — use `query={}`. Cloud-penetrating (no cloud mask needed).
**Landsat 8/9 C2** (`landsat-c2-l2`): 30m (SR_B1–SR_B7), 100m thermal (ST_B10). Scale=0.0000275, offset=-0.2. 16-day revisit.

## Commands

Install the package in development mode:
```bash
pip install -e .
```

## Dependencies

Core: numpy, pandas, xarray, dask, geopandas, rioxarray, stackstac, pystac-client, planetary-computer, pyproj, shapely, rasterio, netcdf4, matplotlib
