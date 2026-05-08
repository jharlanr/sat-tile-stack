"""
I/O utilities for satellite tile timestacks.

Handles NetCDF writing with CF-1.8 convention compliance, attribute
sanitization, and GeoTIFF export.
"""

import xarray as xr
import numpy as np
import json, re, warnings
from datetime import datetime, timezone


# ===========================================================================
# Attribute cleaning helpers
# ===========================================================================

def scrub_attrs(xr_obj, drop=()):
    """Make all attributes JSON-serializable. Drop specified keys."""
    obj = xr_obj.copy()
    for k in list(obj.attrs):
        if k in drop:
            obj.attrs.pop(k)
            continue
        v = obj.attrs[k]
        try:
            json.dumps(v)
        except (TypeError, OverflowError):
            if isinstance(v, np.generic):
                obj.attrs[k] = v.item()
            else:
                obj.attrs[k] = str(v)
    return obj


_illegal_nc_name = re.compile(r"[^0-9A-Za-z_]")

def sanitise_dataset(ds):
    """
    Move variables with illegal NetCDF names or object dtype to global attributes.
    Never touches dimension coordinates (band, time, x, y).
    """
    ds = ds.copy()
    dim_coords = set(ds.dims)
    for v in list(ds.variables):
        if v in dim_coords:
            continue
        bad_name = (":" in v) or _illegal_nc_name.search(v)
        bad_type = ds[v].dtype == object
        if not (bad_name or bad_type):
            continue

        val = ds[v].values
        if val.size == 1:
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.tolist() if val.size < 100 else json.dumps(val.tolist())
        ds = ds.drop_vars(v)
        ds.attrs[v] = val
    return ds


def coerce_attrs_to_json_safe(ds):
    """Final pass: ensure all attributes are JSON-serializable."""
    for k, v in list(ds.attrs.items()):
        if isinstance(v, set):
            if len(v) == 1:
                ds.attrs[k] = next(iter(v))
            else:
                ds.attrs[k] = list(v)
        elif isinstance(v, np.generic):
            ds.attrs[k] = v.item()

        try:
            json.dumps(ds.attrs[k])
        except (TypeError, OverflowError):
            ds.attrs[k] = str(ds.attrs[k])
    return ds


# ===========================================================================
# CF-compliant NetCDF writing
# ===========================================================================

def _add_cf_metadata(ds, da):
    """Backward-compat wrapper around ``finalize_cf`` for single-DataArray flows.

    Pulls a CRS hint from the DataArray's attrs (where stage-1 build keeps it)
    and delegates to the Dataset-level finalizer.
    """
    crs_hint = da.attrs.get("crs") or da.attrs.get("epsg")
    if crs_hint and not ds.attrs.get("crs"):
        ds.attrs["crs"] = str(crs_hint)
    return finalize_cf(ds, spatial_vars=("reflectance",))


# Known variables we annotate when present. Keys map var-name -> attrs dict.
_KNOWN_VAR_ATTRS = {
    "reflectance": {
        "long_name": "Sentinel-2 reflectance and derived per-pixel layers",
        "units": "1",
    },
    "water_area": {
        "long_name": "lake water area",
        "units": "m2",
    },
    "cloudy_seq_rgb": {
        "long_name": "tile usefulness prediction (RGB CNN)",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "cloudy useful",
    },
    "cloudy_seq_rgbn": {
        "long_name": "tile usefulness prediction (RGB+NIR CNN)",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "cloudy useful",
    },
    "cloudy_seq_bns16": {
        "long_name": "tile usefulness prediction (Blue+NIR+SWIR16 CNN)",
        "units": "1",
        "flag_values": np.array([0, 1], dtype=np.int8),
        "flag_meanings": "cloudy useful",
    },
    "eo_cloud_cover": {
        "long_name": "EO cloud-cover percentage from STAC item",
        "units": "percent",
    },
    "pct_nans": {
        "long_name": "percentage of NaN pixels in the full tile",
        "units": "percent",
    },
    "pctnanpix_inmask": {
        "long_name": "percentage of NaN pixels inside the lake mask",
        "units": "1",
    },
    "pctcloudypix_inmask": {
        "long_name": "percentage of cloudy pixels inside the lake mask",
        "units": "1",
    },
}


def finalize_cf(
    ds,
    spatial_vars=None,
    grid_mapping_name="crs",
    source="sat-tile-stack (Microsoft Planetary Computer)",
    title=None,
    history_action="created",
):
    """Add CF-1.8 metadata to a Dataset in place. Idempotent.

    - Sets/normalizes Conventions, title, source; appends a history line.
    - Annotates x/y/time/band coordinate attrs.
    - Builds a scalar ``grid_mapping`` variable from ``ds.attrs["crs"]``
      or ``ds.attrs["proj:code"]`` or the ``epsg`` scalar coord, and
      attaches ``grid_mapping = <name>`` to every spatial data variable.
    - Sets known long_name/units on common variables (water_area,
      cloudy_seq_*, etc.) without clobbering user-provided values.

    Parameters
    ----------
    ds : xarray.Dataset
    spatial_vars : iterable of str, optional
        Data variables that share the (y, x) grid. Auto-detected when None
        as any data var with both 'y' and 'x' in its dims.
    grid_mapping_name : str, default "crs"
        Name of the scalar grid-mapping variable to create / refer to.
    source : str
        Value for the global ``source`` attr (set only if missing).
    title : str, optional
        Value for the global ``title`` attr (set only if missing).
    history_action : str
        Verb to log in the ``history`` attr (e.g. "created", "appended NDWI").
    """
    # Auto-detect spatial vars: anything with both y and x dims.
    if spatial_vars is None:
        spatial_vars = tuple(
            name for name, v in ds.data_vars.items()
            if {"y", "x"}.issubset(set(v.dims))
        )

    # --- Global attrs ---
    existing = ds.attrs.get("Conventions", "")
    if "CF-1" not in str(existing):
        ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs.setdefault("title", title or "Satellite image timestack")
    ds.attrs.setdefault("source", source)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_line = f"{stamp}: {history_action} by sat-tile-stack"
    prior = ds.attrs.get("history")
    ds.attrs["history"] = f"{prior}\n{history_line}" if prior else history_line

    # --- Coordinate attrs ---
    if "x" in ds.coords:
        ds["x"].attrs.update({
            "units": ds["x"].attrs.get("units", "m"),
            "standard_name": "projection_x_coordinate",
            "long_name": ds["x"].attrs.get("long_name", "x coordinate of projection"),
            "axis": "X",
        })
    if "y" in ds.coords:
        ds["y"].attrs.update({
            "units": ds["y"].attrs.get("units", "m"),
            "standard_name": "projection_y_coordinate",
            "long_name": ds["y"].attrs.get("long_name", "y coordinate of projection"),
            "axis": "Y",
        })
    if "time" in ds.coords:
        ds["time"].attrs.update({
            "standard_name": "time",
            "long_name": ds["time"].attrs.get("long_name", "observation time"),
            "axis": "T",
        })
    if "band" in ds.coords:
        ds["band"].attrs.setdefault(
            "long_name", "spectral band or derived layer index"
        )

    # --- Known variable annotations ---
    for vname, vattrs in _KNOWN_VAR_ATTRS.items():
        if vname not in ds.variables:
            continue
        for k, v in vattrs.items():
            ds[vname].attrs.setdefault(k, v)

    # --- Build the grid-mapping variable from a CRS hint ---
    crs_hint = (
        ds.attrs.get("crs")
        or ds.attrs.get("proj:code")
        or (ds["epsg"].item() if "epsg" in ds.coords or "epsg" in ds.variables else None)
    )
    if crs_hint:
        try:
            import pyproj
            crs = pyproj.CRS.from_user_input(crs_hint)
            cf_params = crs.to_cf()
            cf_params["crs_wkt"] = crs.to_wkt()
            epsg = crs.to_epsg()
            if epsg:
                cf_params["epsg_code"] = epsg
            # Replace any existing grid-mapping var to keep attrs fresh.
            ds[grid_mapping_name] = xr.DataArray(np.int32(0), attrs=cf_params)
            for vname in spatial_vars:
                if vname in ds.data_vars:
                    ds[vname].attrs["grid_mapping"] = grid_mapping_name
        except Exception:
            pass

    return ds


def write_netcdf(
    ds,
    outfile,
    spatial_vars=None,
    encoding=None,
    history_action="created",
    finalize=True,
    drop_attrs=("spec",),
):
    """Write a (possibly multi-variable) Dataset to a CF-1.8 NetCDF-4 file.

    Parameters
    ----------
    ds : xarray.Dataset
    outfile : str or Path
    spatial_vars : iterable of str, optional
        Forwarded to ``finalize_cf``.
    encoding : dict, optional
        Per-variable encoding overrides; sensible defaults are filled in
        for spatial vars (zlib level 4, float32) and ``time`` (CF-standard
        units & calendar).
    history_action : str
        Logged in the global ``history`` attr.
    finalize : bool
        If False, skip ``finalize_cf`` (caller has already done it).
    drop_attrs : tuple of str
        Attribute keys scrubbed from the dataset before writing.
    """
    ds = ds.copy()
    ds = scrub_attrs(ds, drop=drop_attrs)
    ds = sanitise_dataset(ds)
    ds = coerce_attrs_to_json_safe(ds)
    if finalize:
        ds = finalize_cf(ds, spatial_vars=spatial_vars,
                         history_action=history_action)

    # Encoding: zlib for spatial vars, CF-standard for time.
    enc = {}
    for vname, v in ds.data_vars.items():
        if {"y", "x"}.issubset(set(v.dims)):
            enc[vname] = dict(zlib=True, complevel=4)
            if np.issubdtype(v.dtype, np.floating):
                enc[vname]["dtype"] = "float32"
    if "time" in ds.variables:
        enc["time"] = dict(units="days since 1970-01-01", calendar="standard")
    if encoding:
        for k, v in encoding.items():
            enc.setdefault(k, {}).update(v)

    ds.to_netcdf(
        outfile,
        engine="netcdf4",
        format="NETCDF4",
        mode="w",
        encoding=enc,
    )
    return outfile


def write_netcdf_from_da(da, outfile, drop_attrs=("spec",)):
    """
    Clean a DataArray and write it to a CF-1.8 compliant NetCDF-4 file.

    Parameters
    ----------
    da : xarray.DataArray
        The timestack to write. Expected dims: (time, band, y, x).
    outfile : str or Path
        Output file path.
    drop_attrs : tuple of str
        Attribute keys to drop before writing (default: ("spec",)).
    """
    # 1. Drop band-indexed STAC metadata coords (e.g., common_name, title,
    #    center_wavelength) — preserve them in attrs as JSON
    band_coords = [k for k, v in da.coords.items()
                   if "band" in v.dims and k != "band"]
    if band_coords:
        stac_meta = {}
        for coord in band_coords:
            vals = da.coords[coord].values
            stac_meta[coord] = {str(b): str(v) for b, v in zip(da.band.values, vals)}
        da = da.drop_vars(band_coords)
        da.attrs["stac_band_metadata"] = json.dumps(stac_meta)

    # 2. Scrub global attrs → JSON-safe
    da_clean = scrub_attrs(da, drop=drop_attrs)

    # 3. Promote to Dataset
    ds = da_clean.to_dataset(name="reflectance", promote_attrs=True)

    # 4. Ensure band coordinate is string-typed (not object) for NetCDF
    if "band" in ds and ds["band"].dtype == object:
        ds["band"] = ds["band"].astype(str)

    # 5. Move illegal / object vars → attrs
    ds = sanitise_dataset(ds)
    ds = coerce_attrs_to_json_safe(ds)

    # 6. Add CF-1.8 metadata
    ds = _add_cf_metadata(ds, da)

    # 7. Encoding
    enc = {
        "reflectance": dict(zlib=True, complevel=4, dtype="float32"),
    }
    # Encode time as CF-standard
    if "time" in ds:
        enc["time"] = dict(units="days since 1970-01-01", calendar="standard")

    # 8. Write
    ds.to_netcdf(
        outfile,
        engine="netcdf4",
        format="NETCDF4",
        mode="w",
        encoding=enc,
    )
    print(f"wrote {outfile}")


# ===========================================================================
# GeoTIFF export
# ===========================================================================

def export_geotiff(da, outfile, time_index=0, bands=None):
    """
    Export a single timestep from a timestack as a georeferenced GeoTIFF.

    The output file can be opened directly in QGIS, ArcGIS, or any GDAL-based tool
    with full spatial referencing (CRS + transform).

    Parameters
    ----------
    da : xarray.DataArray
        Timestack with dims (time, band, y, x) and CRS information.
    outfile : str or Path
        Output GeoTIFF file path.
    time_index : int or str, optional
        Which timestep to export (default: 0).
    bands : list of str, optional
        Which bands to include. If None, includes all bands.
    """
    import rioxarray  # noqa: F401

    if isinstance(time_index, str):
        frame = da.sel(time=time_index)
    else:
        frame = da.isel(time=time_index)

    if bands is not None:
        frame = frame.sel(band=bands)

    if frame.rio.crs is None:
        crs = da.attrs.get("crs") or da.attrs.get("epsg")
        if crs is not None:
            frame = frame.rio.write_crs(crs)
        else:
            warnings.warn(
                "No CRS found on the DataArray. The GeoTIFF will not be georeferenced. "
                "Set CRS with: da = da.rio.write_crs('EPSG:3413')"
            )

    frame.rio.to_raster(str(outfile))
    print(f"Exported GeoTIFF: {outfile} (shape: {frame.shape})")
