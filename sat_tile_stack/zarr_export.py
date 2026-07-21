"""
Export a timestack DataArray to a GeoZarr store for the browser viewer.

The labeling GUI's image panel renders timestacks client-side on the GPU via
deck.gl-zarr (see the `viewer.js` widget). That widget reads a single-resolution
zarr **v3 root array** with dims (time, band, y, x) and GeoZarr metadata on the
array attrs. This module writes exactly that layout from an in-memory
`reflectance` DataArray, so the Flask backend can convert `.nc` timestacks on
demand.

Layout written
--------------
- root array, dtype uint16, shape (time, n_rgb[, +1 mask], y, x)
- dimension_names = ["time", "band", "y", "x"]
- attrs:
    spatial:dimensions   = ["y", "x"]
    spatial:shape        = [H, W]
    spatial:transform    = [dx, 0, x0, 0, dy, y0]  (pixel col/row -> CRS x/y)
    spatial:registration = "pixel"
    proj:wkt2            = WKT2 for the timestack CRS  (offline, no epsg.io)
    sts:bands           = ["B04", "B03", "B02", ("mask")]   (UI convenience)
    sts:dates           = ISO date strings per timestep
    sts:mask_band       = index of the mask band, or -1 if absent
"""

import shutil

import numpy as np
from pyproj import CRS

# Sentinel-2 / Landsat true-color band order for R, G, B.
_RGB_CANDIDATES = [
    ["B04", "B03", "B02"],
    ["SR_B4", "SR_B3", "SR_B2"],
]
_MASK_NAMES = ("mask", "water_mask", "ndwi_mask")


def _band_names(da):
    """Return the list of string band names, from `band_name` aux coord or `band`."""
    if "band_name" in da.coords:
        return [str(b) for b in da["band_name"].values]
    return [str(b) for b in da["band"].values]


def _pick_rgb(names):
    for opt in _RGB_CANDIDATES:
        if all(b in names for b in opt):
            return opt
    # Fallback: first three bands that aren't a mask.
    non_mask = [n for n in names if n not in _MASK_NAMES]
    return non_mask[:3]


def _transform_from_xy(x, y):
    """Affine (a,b,c,d,e,f) from projected cell-center coords, pixel registration."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # north-up -> negative
    x0 = float(x[0]) - dx / 2.0
    y0 = float(y[0]) - dy / 2.0
    return [dx, 0.0, x0, 0.0, dy, y0]


def _crs_wkt2(da):
    hint = da.attrs.get("crs") or da.attrs.get("epsg")
    if hint is None and "spatial_ref" in da.coords:
        hint = da["spatial_ref"].attrs.get("crs_wkt")
    if hint is None:
        raise ValueError(
            "Timestack has no CRS (looked in attrs['crs'], attrs['epsg'], "
            "spatial_ref.crs_wkt). Cannot write a geolocated GeoZarr store."
        )
    return CRS.from_user_input(hint).to_wkt("WKT2_2019")


def da_to_geozarr(da, out_dir, mask_band="mask", overwrite=True):
    """
    Write a (time, band, y, x) reflectance DataArray to a GeoZarr root array.

    Parameters
    ----------
    da : xarray.DataArray
        Dims (time, band, y, x). `band` may be an integer index with a
        `band_name` auxiliary coordinate (as written by sattile_stack), or
        string-labelled directly.
    out_dir : str or pathlib.Path
        Destination `.zarr` directory.
    mask_band : str
        Name of a mask band to include as an extra channel if present.
    overwrite : bool
        Remove an existing store at out_dir first.

    Returns
    -------
    dict
        Summary: {"n_frames", "shape", "bands", "mask_band_index"}.
    """
    import zarr  # local import: only needed when actually exporting

    from pathlib import Path

    out_dir = Path(out_dir)
    da = da.transpose("time", "band", "y", "x")
    names = _band_names(da)
    rgb = _pick_rgb(names)

    # Assemble the output band order: RGB first, then mask (if present).
    out_names = list(rgb)
    mask_idx = -1
    if mask_band in names:
        out_names.append(mask_band)

    # Select the chosen bands by position (band axis may be an int index).
    sel = [names.index(n) for n in out_names]
    data = da.isel(band=sel).values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    if mask_band in names:
        mask_idx = out_names.index(mask_band)
        # Store mask as 0/255 so the viewer can threshold it cheaply.
        data[:, mask_idx] = (data[:, mask_idx] > 0.5) * 255.0

    n_time, n_band, H, W = data.shape
    dates = [str(np.datetime_as_string(t, unit="D")) for t in da.time.values]
    transform = _transform_from_xy(da.x.values, da.y.values)
    wkt2 = _crs_wkt2(da)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.storage.LocalStore(str(out_dir))
    arr = zarr.create_array(
        store=store,
        name="/",
        shape=data.shape,
        chunks=(1, n_band, H, W),  # one chunk per timestep -> one tile
        dtype="uint16",
        dimension_names=["time", "band", "y", "x"],
        # No compression: the browser widget bundles only zarrita's built-in
        # "bytes" codec. zstd/blosc would need separate .wasm files that the
        # single-file IIFE can't load, so a compressed store renders blank.
        # Chunks are raw uint16 (~1.5 MB) served over localhost — size is moot.
        compressors=None,
    )
    arr[:] = np.clip(data, 0, 65535).astype(np.uint16)
    arr.attrs.update(
        {
            "spatial:dimensions": ["y", "x"],
            "spatial:shape": [H, W],
            "spatial:transform": transform,
            "spatial:registration": "pixel",
            "proj:wkt2": wkt2,
            "sts:bands": out_names,
            "sts:dates": dates,
            "sts:mask_band": mask_idx,
        }
    )
    return {
        "n_frames": n_time,
        "shape": [n_time, n_band, H, W],
        "bands": out_names,
        "mask_band_index": mask_idx,
    }
