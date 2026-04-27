"""
I/O utilities for satellite tile timestacks.

Handles NetCDF writing with CF-1.8 convention compliance, attribute
sanitization, and GeoTIFF export.
"""

import xarray as xr
import numpy as np
import json, re, warnings
from datetime import datetime


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
    """
    Add CF-1.8 convention metadata to a Dataset before writing.

    Adds:
    - Global attributes: Conventions, title, history, source
    - Coordinate attributes: units, standard_name, long_name, axis
    - CRS as a grid_mapping variable
    - Data variable attributes: long_name, grid_mapping
    """
    # --- Global attributes ---
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["title"] = ds.attrs.get("title", "Satellite image timestack")
    ds.attrs["source"] = ds.attrs.get("source", "sat-tile-stack (Microsoft Planetary Computer)")
    ds.attrs["history"] = f"Created {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} by sat-tile-stack"

    # --- x coordinate ---
    if "x" in ds.coords:
        ds["x"].attrs.update({
            "units": "m",
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "axis": "X",
        })

    # --- y coordinate ---
    if "y" in ds.coords:
        ds["y"].attrs.update({
            "units": "m",
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "axis": "Y",
        })

    # --- time coordinate ---
    if "time" in ds.coords:
        ds["time"].attrs.update({
            "standard_name": "time",
            "long_name": "observation time",
            "axis": "T",
        })

    # --- band coordinate ---
    if "band" in ds.coords:
        ds["band"].attrs.update({
            "long_name": "spectral band or derived layer",
        })

    # --- reflectance variable ---
    if "reflectance" in ds:
        ds["reflectance"].attrs.update({
            "long_name": "surface reflectance or derived value",
        })

    # --- CRS as grid_mapping variable (CF convention) ---
    crs_str = da.attrs.get("crs") or da.attrs.get("epsg")
    if crs_str:
        try:
            import pyproj
            crs = pyproj.CRS(crs_str)
            cf_params = crs.to_cf()

            # Create a scalar grid_mapping variable
            gm_name = "crs"
            ds[gm_name] = xr.DataArray(
                data=np.int32(0),
                attrs=cf_params,
            )
            # Add EPSG code as an extra attribute
            epsg = crs.to_epsg()
            if epsg:
                ds[gm_name].attrs["epsg_code"] = epsg

            # Link the data variable to the grid mapping
            if "reflectance" in ds:
                ds["reflectance"].attrs["grid_mapping"] = gm_name

        except Exception:
            pass  # if CRS parsing fails, skip grid_mapping

    return ds


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
