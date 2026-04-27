"""
Functions for attaching additional data to an existing timestack.

Supports both in-memory xarray DataArrays and on-disk NetCDF files.
All functions return the modified DataArray (and optionally re-save to disk).

Examples
--------
In-memory:

    >>> stack = append_band(stack, ndvi_array, "NDVI")
    >>> stack = append_timeseries(stack, water_area, "water_area_m2")
    >>> stack = append_metadata(stack, "lake_name", "Lake Hazen")

File-based (reads, modifies, writes back):

    >>> append_band("tstack_lake42.nc", dem_2d, "DEM")
    >>> append_timeseries("tstack_lake42.nc", temps, "air_temp_C")
    >>> append_metadata("tstack_lake42.nc", "source", "ERA5")
"""

import numpy as np
import xarray as xr
from pathlib import Path

from .io import write_netcdf_from_da


def _load_if_path(stack, var="reflectance"):
    """If stack is a file path, open it and return (DataArray, path). Else (stack, None)."""
    if isinstance(stack, (str, Path)):
        path = Path(stack)
        ds = xr.open_dataset(path)
        return ds[var], path
    return stack, None


def _save_if_path(da, path):
    """If path is not None, write back to disk."""
    if path is not None:
        write_netcdf_from_da(da, str(path))
    return da


def append_band(stack, data, name, var="reflectance"):
    """
    Add a spatial band to a timestack.

    Parameters
    ----------
    stack : xarray.DataArray or str/Path
        The timestack [time, band, y, x], or path to a NetCDF file.
    data : xarray.DataArray or numpy.ndarray
        The data to append. Shape must be either:
        - [time, y, x] for a time-varying band
        - [y, x] for a static band (broadcast across all timesteps)
    name : str
        Name for the new band dimension entry.
    var : str
        Variable name in the NetCDF file (default: "reflectance").

    Returns
    -------
    xarray.DataArray
        The timestack with the new band appended.
    """
    da, path = _load_if_path(stack, var=var)

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = xr.DataArray(
                data, dims=("y", "x"),
                coords={"y": da.y, "x": da.x},
            )
        elif data.ndim == 3:
            data = xr.DataArray(
                data, dims=("time", "y", "x"),
                coords={"time": da.time, "y": da.y, "x": da.x},
            )
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    # Broadcast static (y, x) data across time
    if "time" not in data.dims:
        data = data.expand_dims(time=da.time)

    # Shape as (time, band, y, x) with the new band name
    new_band = data.expand_dims(dim={"band": [name]}).transpose("time", "band", "y", "x")

    # Move band-indexed STAC metadata coords (e.g., 'title', 'platform') to
    # attrs before concatenating. They won't exist on the new band, which
    # causes xr.concat to fail. Preserving them in attrs keeps the info
    # accessible without blocking the concat.
    band_coords = [k for k, v in da.coords.items()
                   if "band" in v.dims and k != "band"]
    if band_coords:
        stac_meta = {}
        for coord in band_coords:
            vals = da.coords[coord].values
            # Store as {band_name: value} dict
            stac_meta[coord] = dict(zip(da.band.values, vals))
        da.attrs["stac_band_metadata"] = stac_meta
    da_clean = da.drop_vars(band_coords)

    # Propagate non-band coordinates from original stack
    non_band = {k: v for k, v in da_clean.coords.items() if "band" not in v.dims}
    new_band = new_band.assign_coords(non_band)

    result = xr.concat([da_clean, new_band], dim="band")
    return _save_if_path(result, path)


def append_timeseries(stack, series, name, var="reflectance"):
    """
    Add a 1D time series as a coordinate on the time dimension.

    Parameters
    ----------
    stack : xarray.DataArray or str/Path
        The timestack, or path to a NetCDF file.
    series : array-like
        1D array with length matching stack's time dimension.
    name : str
        Name for the new coordinate.
    var : str
        Variable name in the NetCDF file (default: "reflectance").

    Returns
    -------
    xarray.DataArray
        The timestack with the new time-coordinate attached.
    """
    da, path = _load_if_path(stack, var=var)

    series = np.asarray(series)
    if len(series) != len(da.time):
        raise ValueError(
            f"Series length {len(series)} != time dimension length {len(da.time)}"
        )

    result = da.assign_coords({name: ("time", series)})
    return _save_if_path(result, path)


def append_metadata(stack, key, value, var="reflectance"):
    """
    Add scalar or static metadata to the timestack's attrs.

    Parameters
    ----------
    stack : xarray.DataArray or str/Path
        The timestack, or path to a NetCDF file.
    key : str
        Attribute name.
    value : scalar, str, list, or dict
        Attribute value (must be JSON-serializable for NetCDF storage).
    var : str
        Variable name in the NetCDF file (default: "reflectance").

    Returns
    -------
    xarray.DataArray
        The timestack with the new attribute set.
    """
    da, path = _load_if_path(stack, var=var)
    da.attrs[key] = value
    return _save_if_path(da, path)
