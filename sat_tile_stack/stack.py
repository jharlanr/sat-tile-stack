"""
Satellite tile stacking module.

Provides functionality to build daily multi-band image time-stacks from
Sentinel-2 imagery via Microsoft Planetary Computer.
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.diagnostics
import pyproj
import stackstac
from pystac.extensions.projection import ProjectionExtension as proj

from .bounds import (
    sat_mask_array,
    bounds_latlon_around,
    pctnanpix_inmask,
    pctcloudypix_inmask,
)
from .utils import combo_scaler, cloud_pix_mask


def sattile_stack(
    catalog,
    centroid,
    band_names,
    pix_res=10,
    tile_size=1024,
    time_range="2019-05-01/2019-09-30",
    normalize=True,
    cloudmask=False,
    mask=None,
    pull_to_mem=False,
):
    """
    Generate a daily, multi-band image time-stack centered on a point.

    This function:
      1. Searches a STAC catalog (Sentinel-2 L2A) for all scenes covering
         a square of size (in meters) around `centroid` between the dates
         in `time_range`.
      2. Builds a 4D xarray.DataArray (time, band, y, x) using stackstac.
      3. Resamples to a daily cadence, carrying forward the maximum
         cloud-cover metadata for each day.
      4. Optionally applies a robust per-band normalization (to [0, 1]).
      5. Crops to a square "tile" of `tile_size x tile_size` pixels around
         the centroid.
      6. Optionally computes into memory (with a dask progress bar).

    Parameters
    ----------
    catalog : pystac_client.Client
        STAC catalog client (e.g., Microsoft Planetary Computer).
    centroid : tuple of float
        (longitude, latitude) in decimal degrees of the point of interest.
    band_names : list of str
        Sentinel-2 band names (e.g. ['B04','B03','B02','B08','B11']).
    pix_res : int, optional
        Pixel resolution in meters (default: 10).
    tile_size : int, optional
        Size of the output tile in pixels (square) (default: 1024).
    time_range : str, optional
        ISO8601 date range "YYYY-MM-DD/YYYY-MM-DD" for imagery search
        (default: '2019-05-01/2019-09-30').
    normalize : bool, optional
        If True, apply a robust (median/IQR) normalization to each band
        so values are scaled into [0,1] (default: True).
    cloudmask : bool, optional
        If True, compute a pixel-wise cloud mask and append to DataArray
        (default: False).
    mask : geopandas.GeoDataFrame or None, optional
        If not None, generate a spatial mask from the GeoDataFrame and
        append to DataArray (default: None).
    pull_to_mem : bool, optional
        If True, triggers `timestack.compute()` and returns an in-memory
        xarray.DataArray; otherwise returns a lazy dask-backed DataArray
        (default: False).

    Returns
    -------
    xarray.DataArray
        DataArray of shape (time, band, tile_size, tile_size). Coordinates
        include:
          - time: daily steps over `time_range`
          - band: the requested `band_names`
          - y, x: UTM coordinates of the tile
          - eo_cloud_cover: cloud cover percentage
          - pct_nans: percent of tile that is NaNs

    Notes
    -----
    - Requires `stackstac`, `xarray`, `dask`, and a STAC `catalog` in scope.
    - CRS for reprojection is taken from the first item in the search.
    """
    # Search imagery catalog for items matching location and date range
    bounds_latlon = bounds_latlon_around(*centroid, side_m=pix_res * tile_size * 1.1)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds_latlon,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 100}},
    )
    items = search.item_collection()

    # Create the stack
    stack = stackstac.stack(
        items,
        epsg=proj.ext(items[0]).epsg,
        assets=band_names,
        bounds_latlon=bounds_latlon,
        resolution=pix_res,
        chunksize=(1, 1, 4096, 4096),
        fill_value=np.nan,
    )

    # Mask out black 0.0 pixels as NaNs
    nodata_mask = (stack == 0).all(dim="band")
    stack = stack.where(~nodata_mask)

    # Sample daily, preserving cloud cover in the metadata
    stack_daily = stack.resample(time="D").mean("time", keep_attrs=True)
    cc_da = xr.DataArray(
        np.array([item.properties["eo:cloud_cover"] for item in items]),
        coords={"time": stack.time},
        dims=["time"],
    )
    start, end = time_range.split("/")
    full_days = pd.date_range(start, end, freq="D")
    stack_daily = stack_daily.reindex(time=full_days, fill_value=np.nan)
    daily_cc = cc_da.resample(time="D").max()
    daily_cc = daily_cc.reindex(time=full_days, fill_value=np.nan)
    stack_daily = stack_daily.assign_coords(eo_cloud_cover=daily_cc)

    # Apply normalization if requested
    if normalize:
        stack_rechunk = stack_daily.chunk({"y": -1, "x": -1})
        stack_daily = xr.apply_ufunc(
            combo_scaler,
            stack_rechunk,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"range_max": 1},
            keep_attrs=True,
        )

    # Crop tiles to desired size
    buffer = tile_size * pix_res / 2  # [m]
    x_utm, y_utm = pyproj.Proj(stack.crs)(*centroid)
    timestack = stack_daily.loc[
        ..., y_utm + buffer : y_utm - buffer, x_utm - buffer : x_utm + buffer
    ]

    # Track percent of pixels that are NaN
    nan_counts = timestack.isnull().sum(dim=("band", "y", "x"))
    total = len(band_names) * tile_size * tile_size
    pct_nans = (nan_counts / total) * 100
    timestack = timestack.assign_coords(pct_nans=("time", pct_nans.values))

    # Compute cloud mask if requested
    if cloudmask is True:
        cloud_mask_da = cloud_pix_mask(timestack, mask_method="Williamson2018b")
        cloud_mask_da = cloud_mask_da.expand_dims(dim={"band": ["cloudmask"]})
        timestack = xr.concat([timestack, cloud_mask_da], dim="band")

    # Generate spatial mask if provided
    if mask is not None:
        satmask = sat_mask_array(timestack, mask, feature_id=None)
        timestack = xr.concat([timestack, satmask], dim="band")

    # Track percent of pixels within mask that are NaNs and cloudy
    if mask is not None:
        pct_nan = pctnanpix_inmask(timestack)
        timestack = timestack.assign_coords(pctnanpix_inmask=("time", pct_nan.data))
        pct_cloudy = pctcloudypix_inmask(timestack)
        timestack = timestack.assign_coords(pctcloudypix_inmask=("time", pct_cloudy.data))

    # Convert to float32
    timestack = timestack.astype("float32")

    # Pull stack into memory if requested
    if pull_to_mem:
        print(f"Pulling stack into memory, shape: {timestack.shape}")
        with dask.diagnostics.ProgressBar():
            timestack_mem = timestack.compute()
        print(f"Stack loaded, shape: {timestack_mem.shape}")
        return timestack_mem
    else:
        return timestack
