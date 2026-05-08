"""
Co-register external rasterized binary layers onto per-lake tile stacks.

Currently supports appending NDWI water masks from yearly zarr datacubes
(produced upstream by the supraglacial_lake_id pipeline) onto per-lake
tile stacks. The zarr stores live in EPSG:3413; per-lake stacks are in
their native UTM projection (e.g. EPSG:32622 for CW Greenland tile 22W*).

Per-lake flow:
  1. Read the lake's UTM bbox from x/y coords.
  2. Project the bbox into EPSG:3413 (with a small buffer) and crop the zarr.
  3. For each lake timestep, exact-day match against the zarr time index.
       hit  -> reproject (nearest) onto the lake's UTM grid.
       miss -> zero-fill.
  4. Append as a new band of the existing data variable (default
     ``reflectance``) and extend band-indexed coords by replicating the
     last existing band's metadata.
  5. Write to a parallel output path (never overwrites the input).

Designed as a per-lake function for SLURM array jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyproj
import rasterio.crs
import rasterio.transform
import rasterio.warp
import xarray as xr
from rasterio.enums import Resampling


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_ndwi_mask(
    input_nc,
    zarr_path,
    output_nc,
    var: str = "reflectance",
    band_dim: str = "band",
    new_band_name: str = "ndwi_mask",
    bbox_buffer_m: float = 100.0,
    zarr_var: str = "ndwi_mask",
    zarr_crs: str = "EPSG:3413",
) -> dict:
    """Append a co-registered NDWI mask band to a per-lake tile stack.

    Returns a small dict of summary stats: n_time, n_days_matched,
    n_pixels_water (across the whole cube), bbox_3413.
    """
    input_nc = Path(input_nc)
    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    ds_in = xr.open_dataset(input_nc)
    try:
        da = ds_in[var]
        if da.dims != ("time", band_dim, "y", "x"):
            raise ValueError(
                f"{input_nc.name}: expected dims (time, {band_dim}, y, x), "
                f"got {da.dims}"
            )

        crs_in_str = (
            ds_in.attrs.get("crs")
            or ds_in.attrs.get("proj:code")
            or "EPSG:32622"
        )
        crs_in = pyproj.CRS.from_user_input(crs_in_str)
        crs_zarr = pyproj.CRS.from_user_input(zarr_crs)

        x_coords = ds_in.x.values.astype(np.float64)
        y_coords = ds_in.y.values.astype(np.float64)
        times = ds_in.time.values

        bbox_utm = _coord_bbox(x_coords, y_coords)
        bbox_3413 = _project_bbox(bbox_utm, crs_in, crs_zarr, bbox_buffer_m)

        cropped = _crop_zarr(zarr_path, zarr_var, bbox_3413)
        ndwi_cube, n_matched = _build_ndwi_cube(
            cropped, times, x_coords, y_coords, crs_zarr, crs_in,
        )
        ds_out = _append_band(ds_in, var, band_dim, ndwi_cube, new_band_name)
    finally:
        ds_in.close()

    _write_atomic(ds_out, output_nc, var)

    return {
        "input": str(input_nc),
        "output": str(output_nc),
        "n_time": int(len(times)),
        "n_days_matched": int(n_matched),
        "n_pixels_water": int(ndwi_cube.sum()),
        "bbox_3413": tuple(float(v) for v in bbox_3413),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _coord_bbox(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pixel-edge inclusive bbox (x_min, y_min, x_max, y_max) from coord arrays.

    Coords are assumed to be pixel centers on a regular grid.
    """
    pix_x = float(abs(x[1] - x[0]))
    pix_y = float(abs(y[1] - y[0]))
    return (
        float(x.min()) - pix_x / 2,
        float(y.min()) - pix_y / 2,
        float(x.max()) + pix_x / 2,
        float(y.max()) + pix_y / 2,
    )


def _project_bbox(bbox, src_crs, dst_crs, buffer):
    """Project a bbox between CRSs, sampling all 4 corners and buffering."""
    x_min, y_min, x_max, y_max = bbox
    tx = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    cx, cy = tx.transform(
        [x_min, x_max, x_min, x_max],
        [y_min, y_min, y_max, y_max],
    )
    return (
        float(min(cx)) - buffer,
        float(min(cy)) - buffer,
        float(max(cx)) + buffer,
        float(max(cy)) + buffer,
    )


def _crop_zarr(zarr_path, var, bbox):
    """Load a tiny in-memory crop of the zarr to the given EPSG:3413 bbox.

    Reads the needed arrays (``var``, ``x``, ``y``, ``time``) directly via
    the low-level zarr API and constructs an xarray DataArray. Bypasses
    xarray's zarr engine, which eagerly enumerates the whole group and
    chokes on the ``source_items`` array's fill_value in the upstream
    NDWI stores.
    """
    import zarr

    zarr_path = str(zarr_path)
    var_arr = zarr.open_array(f"{zarr_path}/{var}", mode="r")
    x_arr   = zarr.open_array(f"{zarr_path}/x",   mode="r")
    y_arr   = zarr.open_array(f"{zarr_path}/y",   mode="r")
    t_arr   = zarr.open_array(f"{zarr_path}/time", mode="r")

    x_full = np.asarray(x_arr[:], dtype=np.float64)
    y_full = np.asarray(y_arr[:], dtype=np.float64)
    times  = _decode_zarr_time(np.asarray(t_arr[:]), dict(t_arr.attrs))

    x_min, y_min, x_max, y_max = bbox
    x_mask = (x_full >= x_min) & (x_full <= x_max)
    y_mask = (y_full >= y_min) & (y_full <= y_max)
    x_idx = np.where(x_mask)[0]
    y_idx = np.where(y_mask)[0]

    if x_idx.size == 0 or y_idx.size == 0:
        return xr.DataArray(
            np.zeros((len(times), 0, 0), dtype=np.int8),
            dims=("time", "y", "x"),
            coords={"time": times, "y": np.array([]), "x": np.array([])},
        )

    x_lo, x_hi = int(x_idx.min()), int(x_idx.max()) + 1
    y_lo, y_hi = int(y_idx.min()), int(y_idx.max()) + 1
    data = np.asarray(var_arr[:, y_lo:y_hi, x_lo:x_hi])
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": times,
            "y": y_full[y_lo:y_hi],
            "x": x_full[x_lo:x_hi],
        },
    )


def _decode_zarr_time(raw, attrs):
    """Decode a CF-style time array (days/seconds since X) to datetime64[ns]."""
    units = attrs.get("units", "")
    if "since" not in units:
        return raw.astype("datetime64[ns]")
    step, _, ref = units.partition(" since ")
    step = step.strip().lower()
    ref_dt = np.datetime64(ref.strip().replace(" ", "T"))
    unit_map = {
        "days": "D", "day": "D",
        "hours": "h", "hour": "h",
        "minutes": "m", "minute": "m",
        "seconds": "s", "second": "s",
    }
    np_unit = unit_map.get(step, "D")
    return (ref_dt + raw.astype(np.int64).astype(f"timedelta64[{np_unit}]")).astype("datetime64[ns]")


def _build_ndwi_cube(cropped, times, dst_x, dst_y, src_crs, dst_crs):
    """Build a (time, n_y, n_x) uint8 cube on the lake's UTM grid.

    Reprojects only the days where the lake date matches a zarr date
    exactly; everything else is zero.
    """
    n_time = len(times)
    n_y = len(dst_y)
    n_x = len(dst_x)
    cube = np.zeros((n_time, n_y, n_x), dtype=np.uint8)

    if cropped.size == 0 or len(cropped.x) < 2 or len(cropped.y) < 2:
        return cube, 0

    z_x = cropped.x.values.astype(np.float64)
    z_y = cropped.y.values.astype(np.float64)
    z_pix_x = float(abs(z_x[1] - z_x[0]))
    z_pix_y = float(abs(z_y[1] - z_y[0]))
    src_transform = rasterio.transform.from_origin(
        z_x.min() - z_pix_x / 2,
        z_y.max() + z_pix_y / 2,
        z_pix_x, z_pix_y,
    )

    dst_pix_x = float(abs(dst_x[1] - dst_x[0]))
    dst_pix_y = float(abs(dst_y[1] - dst_y[0]))
    dst_transform = rasterio.transform.from_origin(
        float(dst_x.min()) - dst_pix_x / 2,
        float(dst_y.max()) + dst_pix_y / 2,
        dst_pix_x, dst_pix_y,
    )

    src_rio = rasterio.crs.CRS.from_wkt(src_crs.to_wkt())
    dst_rio = rasterio.crs.CRS.from_wkt(dst_crs.to_wkt())

    z_days = cropped.time.values.astype("datetime64[D]")
    day_to_zidx = {np.datetime64(d): i for i, d in enumerate(z_days)}

    n_matched = 0
    for t_idx, t in enumerate(times):
        day = np.datetime64(np.datetime64(t, "D"))
        zidx = day_to_zidx.get(day)
        if zidx is None:
            continue
        src = np.ascontiguousarray(cropped.isel(time=zidx).values.astype(np.uint8))
        dst = np.zeros((n_y, n_x), dtype=np.uint8)
        rasterio.warp.reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_rio,
            dst_transform=dst_transform,
            dst_crs=dst_rio,
            resampling=Resampling.nearest,
        )
        cube[t_idx] = dst
        n_matched += 1
    return cube, n_matched


def _append_band(ds_in, var, band_dim, ndwi_cube, new_band_name):
    """Return a new dataset with `ndwi_cube` appended as a new band of `var`.

    Extends every 1-D band-indexed coord by replicating the last band's
    value, so the new NDWI band's metadata mirrors the existing 'mask'
    band's metadata (which is itself a placeholder for non-spectral bands).
    Updates the dataset-level ``band`` attr if present.

    Implementation: builds the extended array at the numpy level and
    constructs a fresh DataArray. Avoids xarray.concat's alignment pass,
    which gets confused by the rich set of band-indexed sub-coords on
    sat-tile-stack outputs.
    """
    da = ds_in[var]
    n_bands = da.sizes[band_dim]
    band_axis = da.dims.index(band_dim)

    # Stack along band axis (numpy-level — no alignment magic).
    new_band = ndwi_cube.astype(da.dtype)
    new_band = np.expand_dims(new_band, axis=band_axis)
    extended = np.concatenate([da.values, new_band], axis=band_axis)

    # Build extended coord values: every coord with dims == (band,) gets one
    # extra entry. For string/object coords (`common_name`, `title`, ...) we
    # use ``new_band_name`` so the new band is identifiable rather than a
    # duplicate of the existing 'mask' band's name. For numeric coords
    # (`gsd`, `center_wavelength`, `full_width_half_max`) we replicate the
    # last band's value — already NaN for the 'mask' band, which is the
    # right placeholder for a non-spectral derived layer.
    # Coords are wrapped as (dims, values) tuples so xarray binds them to
    # the right dim instead of inferring a new same-named dim.
    new_coords = {}
    for cname, c in da.coords.items():
        if c.dims == (band_dim,):
            if cname == band_dim:
                next_idx = int(np.asarray(c.values).max()) + 1
                new_coords[cname] = ((band_dim,), np.append(c.values, next_idx))
            elif np.issubdtype(c.dtype, np.character) or c.dtype == object:
                new_coords[cname] = ((band_dim,), np.append(c.values, new_band_name))
            else:
                last_val = c.isel({band_dim: -1}).values
                new_coords[cname] = ((band_dim,), np.append(c.values, last_val))
        elif band_dim not in c.dims:
            new_coords[cname] = c

    new_da = xr.DataArray(
        extended,
        dims=da.dims,
        coords=new_coords,
        attrs=dict(da.attrs),
        name=da.name,
    )

    # Rebuild the dataset, preserving everything else (water_area,
    # cloudy_seq_*, time-indexed coords, attrs).
    other_vars = {k: v for k, v in ds_in.data_vars.items() if k != var}
    ds_out = xr.Dataset(
        data_vars={var: new_da, **other_vars},
        coords={k: v for k, v in ds_in.coords.items() if band_dim not in v.dims},
        attrs=dict(ds_in.attrs),
    )

    band_names = _read_band_names_attr(ds_in.attrs)
    if band_names is not None and len(band_names) == n_bands:
        ds_out.attrs["band"] = band_names + [new_band_name]

    ds_out.attrs.setdefault(
        "ndwi_mask_source",
        "NDWI > 0.3, Sentinel-2 L2A, cloud-filtered 10%, ice-clipped "
        "(NSIDC-0793). Co-registered from EPSG:3413 zarr to lake's UTM "
        "grid by nearest-neighbor reproject; missing days zero-filled.",
    )
    return ds_out


def _read_band_names_attr(attrs):
    """Return the dataset-level band-name list as a Python list, or None."""
    raw = attrs.get("band")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return None


def _write_atomic(ds, output_nc: Path, var: str):
    """Write to a temp sibling and rename, so partial writes never replace good outputs.

    Routes through :func:`sat_tile_stack.io.write_netcdf` so the output is
    CF-1.8 compliant — Conventions / grid_mapping / per-variable units &
    long_names are added in one place rather than re-implemented here.
    """
    from .io import write_netcdf

    tmp = output_nc.with_suffix(output_nc.suffix + ".tmp")
    write_netcdf(
        ds, tmp,
        spatial_vars=(var,),
        history_action="appended ndwi_mask band via coregister.add_ndwi_mask",
    )
    tmp.replace(output_nc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="Per-lake input .nc")
    p.add_argument("--zarr", required=True, help="Year-matching NDWI zarr")
    p.add_argument("--output", required=True, help="Output .nc path")
    p.add_argument("--var", default="reflectance",
                   help="Data variable to extend (default: reflectance)")
    p.add_argument("--band_dim", default="band",
                   help="Band dimension name (default: band)")
    p.add_argument("--new_band_name", default="ndwi_mask",
                   help="Name to use for the new band (default: ndwi_mask)")
    p.add_argument("--bbox_buffer_m", type=float, default=100.0,
                   help="Buffer (m) added when projecting bbox to EPSG:3413")
    args = p.parse_args()

    summary = add_ndwi_mask(
        args.input, args.zarr, args.output,
        var=args.var,
        band_dim=args.band_dim,
        new_band_name=args.new_band_name,
        bbox_buffer_m=args.bbox_buffer_m,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
