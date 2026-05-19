"""
Co-register external data sources onto per-lake v2 tile stacks.

sat-tile-stack is the *only* writer of the canonical per-lake `.nc`. Every
external source is a producer that emits its own intermediate artifact; the
functions here ingest those artifacts and write them onto the stack as
**new top-level CF data variables** (never as extra bands of `reflectance`).

Three ingest paths, one per artifact shape:

  add_raster_zarr    yearly EPSG:3413 binary zarr  -> (time, y, x) uint8
                     e.g. Shahin NDWI daily masks  -> `water_mask_ndwi`
  add_static_polygon GeoJSON lake polygon          -> (y, x)       uint8
                     e.g. Dunmire 2021 footprint   -> `lake_boundary` (STATIC)
  add_scalar_series  (ids, time) NetCDF series     -> (time,)       float32
                     e.g. Dunmire 2025 S2_water    -> `p_water`

All writes go through `io.write_netcdf` (CF-1.8 finalize + atomic temp-rename)
so partial writes can't corrupt a good file and grid_mapping is consistent.
Designed as per-lake functions for SLURM array jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio.crs
import rasterio.features
import rasterio.transform
import rasterio.warp
import xarray as xr
from rasterio.enums import Resampling


# ---------------------------------------------------------------------------
# Default CF attrs for the known ESSD targets (callers may override)
# ---------------------------------------------------------------------------

NDWI_MASK_ATTRS = {
    "long_name": "NDWI-derived supraglacial water mask (dynamic)",
    "flag_values": np.array([0, 1], dtype=np.uint8),
    "flag_meanings": "no_water water",
    "source": (
        "supraglacial_lake_id pipeline (Shahin): Sentinel-2 L2A NDWI > 0.3 "
        "daily binary mask, ice-clipped (NSIDC-0793). Reprojected from "
        "EPSG:3413 to the lake UTM grid by nearest-neighbour. A stack day "
        "with no matching NDWI date is set to the _FillValue (no NDWI "
        "observation), decoded as NaN on read - it is NOT no_water."
    ),
}

LAKE_BOUNDARY_ATTRS = {
    "long_name": "static lake footprint (Dunmire 2021)",
    "flag_values": np.array([0, 1], dtype=np.uint8),
    "flag_meanings": "outside_lake inside_lake",
    "source": (
        "Dunmire et al. (2021) per-lake maximum-extent polygon, rasterized "
        "onto the lake UTM grid (all_touched). Time-invariant."
    ),
}

P_WATER_ATTRS = {
    "long_name": "fractional lake water extent (Dunmire 2025, S2_water)",
    "units": "1",
    "valid_range": np.array([0.0, 1.0], dtype="float32"),
    "comment": (
        "Dimensionless S2-derived water fraction in [0, 1] (NOT an area). "
        "Aligned to stack dates by calendar day; NaN where Dunmire has no "
        "value for that day."
    ),
    "source": "Dunmire et al. (2025), S2_water.",
}

NDWI_FROM_STACK_ATTRS = {
    "long_name": "NDWI-derived supraglacial water mask",
    "flag_values": np.array([0, 1], dtype=np.uint8),
    "flag_meanings": "no_water water",
    "source": (
        "Water where NDWI = (B02 - B04)/(B02 + B04) > 0.3, computed "
        "pixel-wise from this dataset's own Sentinel-2 L2A surface "
        "reflectance (boa_add_offset applied per timestep). No cloud masking "
        "and no ice-sheet clip applied - combine with the `cloud_mask` "
        "variable downstream as needed. NDWI index and threshold after "
        "Dunmire et al. (2021, 2025)."
    ),
    "references": "Dunmire et al. (2021); Dunmire et al. (2025)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_raster_zarr(
    input_nc,
    output_nc,
    *,
    zarr_path,
    zarr_var: str = "ndwi_mask",
    target_var: str = "water_mask_ndwi",
    zarr_crs: str = "EPSG:3413",
    bbox_buffer_m: float = 100.0,
    target_dtype="uint8",
    target_attrs: dict | None = None,
    nodata: int = 255,
    nan_gate_var: str | None = None,
) -> dict:
    """Reproject a yearly EPSG:3413 binary zarr onto the lake grid as a new
    ``(time, y, x)`` data variable. Exact-day match; a stack day with no
    matching zarr date is filled with ``nodata`` and declared as the
    CF ``_FillValue`` (decoded to NaN on read - missing, not no_water).

    ``nan_gate_var`` (e.g. ``"p_water"``) is an existing ``(time,)`` variable
    in *input_nc* used as a cross-product observation gate: any timestep where
    it is NaN is also set to ``nodata`` here. Rationale: Shahin's merged-daily
    product cannot distinguish 'this lake's tile unimaged' from a real 0
    (fill_value 0, source_items unreadable), so an independent per-lake
    'no usable optical observation' signal (Dunmire 2025 p_water missing) is
    used to mark those timesteps missing rather than fabricating no_water.
    Requires the gate variable to already be present -> run add_scalar_series
    before add_raster_zarr.
    """
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    if target_attrs is None:
        target_attrs = NDWI_MASK_ATTRS if target_var == "water_mask_ndwi" else {}

    ds_in = xr.open_dataset(input_nc).load()
    try:
        crs_in = _stack_crs(ds_in)
        crs_zarr = pyproj.CRS.from_user_input(zarr_crs)
        x = ds_in.x.values.astype(np.float64)
        y = ds_in.y.values.astype(np.float64)
        times = ds_in.time.values

        bbox_utm = _coord_bbox(x, y)
        bbox_3413 = _project_bbox(bbox_utm, crs_in, crs_zarr, bbox_buffer_m)
        cropped = _crop_zarr(zarr_path, zarr_var, bbox_3413)
        cube, n_matched = _build_raster_cube(
            cropped, times, x, y, crs_zarr, crs_in, nodata
        )

        # Cross-product observation gate: timesteps where `nan_gate_var`
        # (e.g. p_water) is missing -> this lake had no usable optical
        # observation that day, so the NDWI 0/1 there is not trustworthy.
        # Set those whole frames to `nodata` while still uint8.
        n_gated = 0
        attrs = dict(target_attrs or {})
        if nan_gate_var is not None:
            if nan_gate_var not in ds_in.variables:
                raise KeyError(
                    f"nan_gate_var {nan_gate_var!r} not in {input_nc.name}; "
                    f"run add_scalar_series before add_raster_zarr."
                )
            g = np.asarray(ds_in[nan_gate_var].values)
            if g.shape != (cube.shape[0],):
                raise ValueError(
                    f"{nan_gate_var!r} must be 1-D over time "
                    f"({cube.shape[0]},), got {g.shape}"
                )
            gate = ~np.isfinite(g)
            cube[gate] = nodata
            n_gated = int(gate.sum())
            attrs["comment"] = (
                (attrs.get("comment", "") + f" Timesteps where "
                 f"'{nan_gate_var}' is missing are set to _FillValue "
                 "(no usable optical observation of this lake; cross-product "
                 "gate) even if the NDWI product reported detections.").strip()
            )

        ds_out = _set_data_var(
            ds_in, cube.astype(target_dtype), ("time", "y", "x"),
            target_var, attrs,
        )
        # CF missing-data: declare the no-NDWI-day sentinel as _FillValue so
        # xarray decodes it to NaN on read (consistent with reflectance /
        # p_water no-data). Kept in encoding, not attrs.
        ds_out[target_var].encoding["_FillValue"] = \
            np.dtype(target_dtype).type(nodata)
    finally:
        ds_in.close()

    _write_atomic(ds_out, output_nc,
                  f"co-registered {target_var} from raster zarr")
    return {
        "input": str(input_nc), "output": str(output_nc),
        "target_var": target_var, "n_time": int(len(times)),
        "n_days_matched": int(n_matched),
        "n_days_gated": int(n_gated),
        "n_pixels_water": int((np.asarray(cube) == 1).sum()),
        "bbox_3413": tuple(float(v) for v in bbox_3413),
    }


def add_ndwi_mask(input_nc, zarr_path, output_nc, **kw) -> dict:
    """Backward-compat wrapper: Shahin NDWI zarr -> ``water_mask_ndwi``."""
    return add_raster_zarr(
        input_nc, output_nc, zarr_path=zarr_path,
        zarr_var=kw.pop("zarr_var", "ndwi_mask"),
        target_var=kw.pop("target_var", "water_mask_ndwi"),
        **kw,
    )


def add_ndwi_from_stack(
    input_nc,
    output_nc,
    *,
    ndwi_min: float = 0.3,
    target_var: str = "water_mask_ndwi",
    blue_band: str = "B02",
    red_band: str = "B04",
    nodata: int = 255,
    target_attrs: dict | None = None,
) -> dict:
    """Derive ``water_mask_ndwi`` pixel-wise from the stack's OWN reflectance.

    ``NDWI = (blue - red) / (blue + red)`` on offset-corrected L2A
    reflectance (the /quantification scale cancels in the ratio; the
    additive ``boa_add_offset`` does not, so it is applied per timestep).
    Water where ``NDWI > ndwi_min``. Any non-finite pixel — no-observation
    NaNs already in ``reflectance``, the all-band-zero nodata NaNs, or a
    zero denominator — propagates to ``nodata`` and is declared the CF
    ``_FillValue`` (decodes to NaN on read), strictly **per pixel**, no
    whole-frame logic. No cloud masking, no ice clip (ship `cloud_mask`
    alongside; the user filters cloud downstream). Replaces the external
    NDWI-zarr path; ``add_raster_zarr`` is kept for generic rasters but is
    not used for NDWI.
    """
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    if target_attrs is None:
        target_attrs = (NDWI_FROM_STACK_ATTRS
                        if target_var == "water_mask_ndwi" else {})

    ds_in = xr.open_dataset(input_nc).load()
    try:
        refl = ds_in["reflectance"]
        # band is now a numeric CF index; short names live in `band_name`
        # (fall back to the band coord for legacy string-band files).
        # band_name may round-trip as a data var rather than a coord, so
        # look in .variables (covers both); fall back to legacy string band.
        if "band_name" in ds_in.variables:
            names = [str(b) for b in ds_in["band_name"].values]
        else:
            names = [str(b) for b in refl["band"].values]
        for bn in (blue_band, red_band):
            if bn not in names:
                raise KeyError(f"{bn!r} not in reflectance bands {names}")
        bi, ri = names.index(blue_band), names.index(red_band)

        blue = refl.isel(band=bi).values.astype("float64")
        red = refl.isel(band=ri).values.astype("float64")

        if "boa_add_offset" in ds_in.variables:
            off = np.asarray(ds_in["boa_add_offset"].values,
                             dtype="float64").reshape(-1, 1, 1)
        else:
            off = 0.0
        blue = blue + off          # NaN offset (no-obs day) -> NaN, picked up
        red = red + off

        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = (blue - red) / (blue + red)

        cube = np.full(ndwi.shape, nodata, dtype=np.uint8)
        valid = np.isfinite(ndwi)
        cube[valid] = (ndwi[valid] > ndwi_min).astype(np.uint8)

        ds_out = _set_data_var(
            ds_in, cube, ("time", "y", "x"), target_var, target_attrs
        )
        ds_out[target_var].encoding["_FillValue"] = np.uint8(nodata)
        n_water = int((cube == 1).sum())
        n_missing = int((cube == nodata).sum())
    finally:
        ds_in.close()

    _write_atomic(ds_out, output_nc,
                  f"derived {target_var} = NDWI>{ndwi_min} from stack bands")
    return {
        "input": str(input_nc), "output": str(output_nc),
        "target_var": target_var, "ndwi_min": float(ndwi_min),
        "n_pixels_water": n_water, "n_pixels_missing": n_missing,
        "n_pixels_total": int(cube.size),
    }


def add_static_polygon(
    input_nc,
    output_nc,
    *,
    geojson,
    lake_id: str,
    target_var: str = "lake_boundary",
    id_field: str = "new_id",
    geojson_crs: str = "EPSG:4326",
    all_touched: bool = True,
    target_attrs: dict | None = None,
) -> dict:
    """Rasterize one lake's polygon from a GeoJSON onto the lake grid as a
    **static** ``(y, x)`` uint8 data variable (no time dimension).
    """
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    if target_attrs is None:
        target_attrs = (LAKE_BOUNDARY_ATTRS
                        if target_var == "lake_boundary" else {})

    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    feats = json.load(open(geojson))["features"]
    match = [f for f in feats if str(f["properties"].get(id_field)) == str(lake_id)]
    if not match:
        raise KeyError(
            f"{lake_id!r} not found in {Path(geojson).name} "
            f"(field {id_field!r})"
        )
    geom = shape(match[0]["geometry"])

    ds_in = xr.open_dataset(input_nc).load()
    try:
        crs_in = _stack_crs(ds_in)
        x = ds_in.x.values.astype(np.float64)
        y = ds_in.y.values.astype(np.float64)

        tx = pyproj.Transformer.from_crs(
            pyproj.CRS.from_user_input(geojson_crs), crs_in, always_xy=True
        )
        geom_utm = shp_transform(lambda a, b: tx.transform(a, b), geom)

        dx = float(abs(x[1] - x[0]))
        dy = float(abs(y[1] - y[0]))
        north = float(y.max()) + dy / 2.0
        west = float(x.min()) - dx / 2.0
        transform = rasterio.transform.from_origin(west, north, dx, dy)

        mask = rasterio.features.rasterize(
            [(geom_utm, 1)],
            out_shape=(len(y), len(x)),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
        # rasterize is north-up (row 0 = max y). Match the stack's y order.
        if y[0] < y[-1]:                       # ascending y -> flip rows
            mask = mask[::-1, :]

        ds_out = _set_data_var(
            ds_in, mask, ("y", "x"), target_var, target_attrs
        )
    finally:
        ds_in.close()

    _write_atomic(ds_out, output_nc,
                  f"co-registered static {target_var} from polygon")
    return {
        "input": str(input_nc), "output": str(output_nc),
        "target_var": target_var, "lake_id": str(lake_id),
        "n_pixels_inside": int(mask.sum()),
    }


def add_scalar_series(
    input_nc,
    output_nc,
    *,
    source_nc,
    source_var: str,
    lake_id: str,
    target_var: str = "p_water",
    ids_dim: str = "ids",
    time_dim: str = "time",
    target_attrs: dict | None = None,
) -> dict:
    """Join a ``(ids, time)`` NetCDF series (e.g. Dunmire S2_water) onto the
    lake stack as a ``(time,)`` float32 variable, aligned by calendar day.
    """
    input_nc, output_nc = Path(input_nc), Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    if target_attrs is None:
        target_attrs = P_WATER_ATTRS if target_var == "p_water" else {}

    ds_in = xr.open_dataset(input_nc).load()
    src = xr.open_dataset(source_nc)
    try:
        ids = src[ids_dim].values.astype(str)
        if str(lake_id) not in set(ids):
            raise KeyError(
                f"{lake_id!r} not in {Path(source_nc).name} {ids_dim!r} "
                f"(n={ids.size})"
            )
        series = src[source_var].sel({ids_dim: str(lake_id)})

        src_idx = pd.to_datetime(src[time_dim].values).normalize()
        tgt_idx = pd.to_datetime(ds_in.time.values).normalize()
        s = pd.Series(np.asarray(series.values, dtype="float64"), index=src_idx)
        s = s[~s.index.duplicated(keep="first")]
        vals = s.reindex(tgt_idx).to_numpy().astype("float32")
        n_match = int(np.isfinite(vals).sum())

        ds_out = _set_data_var(
            ds_in, vals, ("time",), target_var, target_attrs
        )
    finally:
        ds_in.close()
        src.close()

    _write_atomic(ds_out, output_nc,
                  f"co-registered {target_var} from scalar series")
    return {
        "input": str(input_nc), "output": str(output_nc),
        "target_var": target_var, "lake_id": str(lake_id),
        "n_time": int(len(vals)), "n_days_matched": n_match,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _stack_crs(ds):
    """Resolve the per-lake stack CRS from attrs, with a CW-Greenland default."""
    hint = (ds.attrs.get("crs") or ds.attrs.get("proj:code")
            or (f"EPSG:{int(ds['epsg'].item())}"
                if "epsg" in ds.variables else None)
            or "EPSG:32622")
    return pyproj.CRS.from_user_input(hint)


def _coord_bbox(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pixel-edge-inclusive (x_min,y_min,x_max,y_max) from pixel-center coords."""
    px = float(abs(x[1] - x[0]))
    py = float(abs(y[1] - y[0]))
    return (float(x.min()) - px / 2, float(y.min()) - py / 2,
            float(x.max()) + px / 2, float(y.max()) + py / 2)


def _project_bbox(bbox, src_crs, dst_crs, buffer):
    """Project a bbox between CRSs, sampling all 4 corners, then buffer."""
    x_min, y_min, x_max, y_max = bbox
    tx = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    cx, cy = tx.transform([x_min, x_max, x_min, x_max],
                          [y_min, y_min, y_max, y_max])
    return (float(min(cx)) - buffer, float(min(cy)) - buffer,
            float(max(cx)) + buffer, float(max(cy)) + buffer)


def _decode_zarr_time(raw, attrs):
    """Decode a CF-style time array (X since ref) to datetime64[ns]."""
    units = attrs.get("units", "")
    if "since" not in units:
        return raw.astype("datetime64[ns]")
    step, _, ref = units.partition(" since ")
    ref_dt = np.datetime64(ref.strip().replace(" ", "T"))
    np_unit = {"days": "D", "day": "D", "hours": "h", "hour": "h",
               "minutes": "m", "minute": "m", "seconds": "s",
               "second": "s"}.get(step.strip().lower(), "D")
    return (ref_dt + raw.astype(np.int64).astype(
        f"timedelta64[{np_unit}]")).astype("datetime64[ns]")


def _crop_zarr(zarr_path, var, bbox):
    """Tiny in-memory crop of the zarr to an EPSG:3413 bbox via low-level API
    (bypasses xarray's eager group enumeration on the upstream NDWI stores).
    """
    import zarr

    zarr_path = str(zarr_path)
    var_arr = zarr.open_array(f"{zarr_path}/{var}", mode="r")
    x_full = np.asarray(zarr.open_array(f"{zarr_path}/x", mode="r")[:], np.float64)
    y_full = np.asarray(zarr.open_array(f"{zarr_path}/y", mode="r")[:], np.float64)
    t_arr = zarr.open_array(f"{zarr_path}/time", mode="r")
    times = _decode_zarr_time(np.asarray(t_arr[:]), dict(t_arr.attrs))

    x_min, y_min, x_max, y_max = bbox
    xi = np.where((x_full >= x_min) & (x_full <= x_max))[0]
    yi = np.where((y_full >= y_min) & (y_full <= y_max))[0]
    if xi.size == 0 or yi.size == 0:
        return xr.DataArray(
            np.zeros((len(times), 0, 0), np.int8),
            dims=("time", "y", "x"),
            coords={"time": times, "y": np.array([]), "x": np.array([])},
        )
    x_lo, x_hi = int(xi.min()), int(xi.max()) + 1
    y_lo, y_hi = int(yi.min()), int(yi.max()) + 1
    data = np.asarray(var_arr[:, y_lo:y_hi, x_lo:x_hi])
    return xr.DataArray(
        data, dims=("time", "y", "x"),
        coords={"time": times, "y": y_full[y_lo:y_hi], "x": x_full[x_lo:x_hi]},
    )


def _build_raster_cube(cropped, times, dst_x, dst_y, src_crs, dst_crs,
                       nodata):
    """(time, ny, nx) cube on the lake UTM grid; only exact-day matches are
    reprojected (nearest). Days with no matching zarr date stay ``nodata``
    (the CF _FillValue), i.e. missing — distinct from a real 0 (no_water).
    """
    n_y, n_x = len(dst_y), len(dst_x)
    cube = np.full((len(times), n_y, n_x), nodata, dtype=np.uint8)
    if cropped.size == 0 or len(cropped.x) < 2 or len(cropped.y) < 2:
        return cube, 0

    zx = cropped.x.values.astype(np.float64)
    zy = cropped.y.values.astype(np.float64)
    zpx, zpy = float(abs(zx[1] - zx[0])), float(abs(zy[1] - zy[0]))
    src_t = rasterio.transform.from_origin(
        zx.min() - zpx / 2, zy.max() + zpy / 2, zpx, zpy)

    dpx, dpy = float(abs(dst_x[1] - dst_x[0])), float(abs(dst_y[1] - dst_y[0]))
    dst_t = rasterio.transform.from_origin(
        float(dst_x.min()) - dpx / 2, float(dst_y.max()) + dpy / 2, dpx, dpy)

    src_rio = rasterio.crs.CRS.from_wkt(src_crs.to_wkt())
    dst_rio = rasterio.crs.CRS.from_wkt(dst_crs.to_wkt())

    day_to_zidx = {np.datetime64(d): i for i, d in
                   enumerate(cropped.time.values.astype("datetime64[D]"))}
    n_matched = 0
    for ti, t in enumerate(times):
        zidx = day_to_zidx.get(np.datetime64(np.datetime64(t, "D")))
        if zidx is None:
            continue
        src = np.ascontiguousarray(
            cropped.isel(time=zidx).values.astype(np.uint8))
        dst = np.zeros((n_y, n_x), np.uint8)
        rasterio.warp.reproject(
            source=src, destination=dst,
            src_transform=src_t, src_crs=src_rio,
            dst_transform=dst_t, dst_crs=dst_rio,
            resampling=Resampling.nearest,
        )
        cube[ti] = dst
        n_matched += 1
    return cube, n_matched


def _set_data_var(ds_in, data, dims, target_var, target_attrs):
    """Return ds_in + a new top-level data variable, preserving everything
    else. Refuses to clobber `reflectance`/`cloud_mask`.
    """
    if target_var in ("reflectance", "cloud_mask"):
        raise ValueError(f"refusing to overwrite core variable {target_var!r}")
    coords = {d: ds_in[d] for d in dims}
    da = xr.DataArray(np.asarray(data), dims=dims, coords=coords,
                      name=target_var, attrs=dict(target_attrs or {}))
    ds_out = ds_in.copy()
    ds_out[target_var] = da
    return ds_out


def _write_atomic(ds, output_nc: Path, history_action: str):
    """Write via io.write_netcdf to a temp sibling, then atomic rename.
    finalize_cf auto-detects spatial vars (everything with y & x dims) so
    grid_mapping is attached consistently; (time,)-only vars are left alone.
    """
    from .io import write_netcdf

    tmp = output_nc.with_suffix(output_nc.suffix + ".tmp")
    write_netcdf(ds, tmp, history_action=history_action)
    Path(tmp).replace(output_nc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main_zarr(argv=None):
    p = argparse.ArgumentParser(description="co-register a raster zarr")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--zarr", required=True)
    p.add_argument("--zarr-var", default="ndwi_mask")
    p.add_argument("--target-var", default="water_mask_ndwi")
    p.add_argument("--bbox-buffer-m", type=float, default=100.0)
    p.add_argument("--nan-gate-var", default=None,
                   help="(time,) var already in --input; its NaN timesteps "
                        "are set missing (e.g. p_water)")
    a = p.parse_args(argv)
    print(json.dumps(add_raster_zarr(
        a.input, a.output, zarr_path=a.zarr, zarr_var=a.zarr_var,
        target_var=a.target_var, bbox_buffer_m=a.bbox_buffer_m,
        nan_gate_var=a.nan_gate_var), indent=2))


def _main_ndwi(argv=None):
    p = argparse.ArgumentParser(
        description="derive water_mask_ndwi from the stack's own bands")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ndwi-min", type=float, default=0.3)
    p.add_argument("--target-var", default="water_mask_ndwi")
    a = p.parse_args(argv)
    print(json.dumps(add_ndwi_from_stack(
        a.input, a.output, ndwi_min=a.ndwi_min,
        target_var=a.target_var), indent=2))


def _main_polygon(argv=None):
    p = argparse.ArgumentParser(description="co-register a static polygon")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--geojson", required=True)
    p.add_argument("--lake-id", required=True)
    p.add_argument("--target-var", default="lake_boundary")
    p.add_argument("--id-field", default="new_id")
    a = p.parse_args(argv)
    print(json.dumps(add_static_polygon(
        a.input, a.output, geojson=a.geojson, lake_id=a.lake_id,
        target_var=a.target_var, id_field=a.id_field), indent=2))


def _main_scalar(argv=None):
    p = argparse.ArgumentParser(description="co-register a scalar series")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--source-var", required=True)
    p.add_argument("--lake-id", required=True)
    p.add_argument("--target-var", default="p_water")
    a = p.parse_args(argv)
    print(json.dumps(add_scalar_series(
        a.input, a.output, source_nc=a.source, source_var=a.source_var,
        lake_id=a.lake_id, target_var=a.target_var), indent=2))


if __name__ == "__main__":
    import sys

    sub = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if sub == "zarr":
        _main_zarr(rest)
    elif sub == "ndwi":
        _main_ndwi(rest)
    elif sub == "polygon":
        _main_polygon(rest)
    elif sub == "scalar":
        _main_scalar(rest)
    else:
        print("usage: python -m sat_tile_stack.coregister "
              "{zarr|ndwi|polygon|scalar} --help")
