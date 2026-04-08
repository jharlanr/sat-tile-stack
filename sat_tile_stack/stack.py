"""
Satellite tile stacking module.

Provides functionality to build multi-band image time-stacks from
satellite imagery via STAC catalogs (e.g., Microsoft Planetary Computer,
Element84 Earth Search).

Supports any STAC collection: Sentinel-2, Sentinel-1, Landsat, etc.
"""

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.diagnostics
import pyproj
import stackstac

# Native resolutions (meters) for known STAC collections.
# Used to warn when pix_res is finer than the sensor supports.
NATIVE_RESOLUTIONS = {
    "sentinel-2-l2a": 10,
    "sentinel-1-grd": 20,  # pixel spacing is 10m but true resolution ~20m
    "landsat-c2-l2": 30,
}

from .bounds import (
    sat_mask_array,
    bounds_latlon_around,
    best_crs_for_point,
    pctnanpix_inmask,
    pctcloudypix_inmask,
)
from .utils import combo_scaler, cloud_pix_mask


def sattile_stack(
    catalog,
    centroid,
    band_names,
    collection="sentinel-2-l2a",
    pix_res=10,
    tile_size=1024,
    time_range="2019-05-01/2019-09-30",
    cadence="D",
    aggregation="mean",
    normalize=True,
    cloudmask=False,
    query=None,
    mask=None,
    pull_to_mem=False,
):
    """
    Generate a multi-band image time-stack centered on a point.

    This function:
      1. Searches a STAC catalog for all scenes in `collection` covering
         a square around `centroid` between the dates in `time_range`.
      2. Builds a 4D xarray.DataArray (time, band, y, x) using stackstac.
      3. Resamples to the requested temporal `cadence`, carrying forward
         cloud-cover metadata when available.
      4. Optionally applies a robust per-band normalization (to [0, 1]).
      5. Crops to a square "tile" of `tile_size x tile_size` pixels around
         the centroid.
      6. Optionally computes into memory (with a dask progress bar).

    Parameters
    ----------
    catalog : pystac_client.Client
        STAC catalog client (e.g., Planetary Computer, Element84).
    centroid : tuple of float
        (longitude, latitude) in decimal degrees of the point of interest.
    band_names : list of str
        Band/asset names for the target collection
        (e.g. ['B04','B03','B02'] for S2, ['VV','VH'] for S1,
        ['SR_B4','SR_B3','SR_B2'] for Landsat).
    collection : str, optional
        STAC collection ID to search (default: 'sentinel-2-l2a').
    pix_res : int, optional
        Pixel resolution in meters (default: 10).
    tile_size : int, optional
        Size of the output tile in pixels (square) (default: 1024).
    time_range : str, optional
        ISO8601 date range "YYYY-MM-DD/YYYY-MM-DD" for imagery search
        (default: '2019-05-01/2019-09-30').
    cadence : str, optional
        Temporal resampling frequency as a pandas offset alias.
        Examples: 'D' (daily), '2D' (every 2 days), 'W' (weekly),
        'MS' (month start), 'ME' (month end). Default: 'D'.
    aggregation : str, optional
        How to combine multiple observations within each cadence window.
        Options:
        - 'mean': average all observations (default). Can blend scenes.
        - 'nearest': pick the observation closest to each cadence timestep.
          Preserves individual scenes with no blending.
        - 'first': take the first observation in each window.
        - 'last': take the last observation in each window.
    normalize : bool, optional
        If True, apply a robust (median/IQR) normalization to each band
        so values are scaled into [0,1] (default: True).
    cloudmask : str, callable, or False, optional
        Cloud masking strategy. Options:
        - False: no cloud mask (default)
        - 'scl': Sentinel-2 Scene Classification Layer (requires 'SCL' band).
          Flags cloud shadow, medium/high probability cloud, and thin cirrus.
        - 'williamson': SWIR1 threshold method (requires 'B11' band).
        - callable: custom function that takes a timestack DataArray and
          returns a DataArray of shape (time, y, x) with 0=clear, 1=cloudy.
    query : dict, optional
        STAC query filter. Default: {"eo:cloud_cover": {"lt": 100}} for
        optical collections. Set to {} to disable filtering.
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
          - time: resampled steps over `time_range` at the given `cadence`
          - band: the requested `band_names`
          - y, x: projected coordinates of the tile
          - eo_cloud_cover: cloud cover % (if available in collection metadata)
          - pct_nans: percent of tile that is NaNs

    Notes
    -----
    - Requires `stackstac`, `xarray`, `dask`, and a STAC `catalog` in scope.
    - CRS for reprojection is taken from the first item's projection extension,
      falling back to best_crs_for_point() if unavailable.
    """
    # Search imagery catalog for items matching location and date range
    bounds_latlon = bounds_latlon_around(*centroid, side_m=pix_res * tile_size * 1.1)

    search_kwargs = {
        "collections": [collection],
        "bbox": bounds_latlon,
        "datetime": time_range,
    }
    if query is not None:
        search_kwargs["query"] = query
    else:
        search_kwargs["query"] = {"eo:cloud_cover": {"lt": 100}}

    try:
        search = catalog.search(**search_kwargs)
        items = search.item_collection()
    except Exception:
        # If cloud_cover query fails (e.g., SAR collection), retry without it
        if query is None:
            search_kwargs.pop("query", None)
            search = catalog.search(**search_kwargs)
            items = search.item_collection()
        else:
            raise

    if len(items) == 0:
        raise ValueError(
            f"No items found for collection='{collection}', "
            f"centroid={centroid}, time_range='{time_range}'"
        )

    # Warn if requested resolution is finer than native
    if collection in NATIVE_RESOLUTIONS and pix_res < NATIVE_RESOLUTIONS[collection]:
        native = NATIVE_RESOLUTIONS[collection]
        print(
            f"WARNING: Requested pix_res={pix_res}m is finer than the native "
            f"resolution of {collection} ({native}m). The output will be "
            f"interpolated — no additional detail beyond {native}m."
        )

    # Determine CRS: prefer projection extension, fall back to best_crs_for_point
    try:
        from pystac.extensions.projection import ProjectionExtension as proj_ext
        epsg = proj_ext.ext(items[0]).epsg
    except Exception:
        crs = best_crs_for_point(*centroid)
        epsg = crs.to_epsg()

    # Create the stack
    stack = stackstac.stack(
        items,
        epsg=epsg,
        assets=band_names,
        bounds_latlon=bounds_latlon,
        resolution=pix_res,
        chunksize=(1, 1, 4096, 4096),
        fill_value=np.nan,
    )

    # Mask out black 0.0 pixels as NaNs
    nodata_mask = (stack == 0).all(dim="band")
    stack = stack.where(~nodata_mask)

    # Resample to requested cadence
    start, end = time_range.split("/")
    full_steps = pd.date_range(start, end, freq=cadence)

    if aggregation == "mean":
        stack_resampled = stack.resample(time=cadence).mean("time", keep_attrs=True)
    elif aggregation == "nearest":
        stack_resampled = stack.reindex(time=full_steps, method="nearest", tolerance=cadence)
    elif aggregation == "first":
        stack_resampled = stack.resample(time=cadence).first(keep_attrs=True)
    elif aggregation == "last":
        stack_resampled = stack.resample(time=cadence).last(keep_attrs=True)
    else:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Supported: 'mean', 'nearest', 'first', 'last'"
        )

    stack_resampled = stack_resampled.reindex(time=full_steps, fill_value=np.nan)

    # Extract cloud cover metadata if available
    if items and "eo:cloud_cover" in items[0].properties:
        cc_da = xr.DataArray(
            np.array([item.properties["eo:cloud_cover"] for item in items]),
            coords={"time": stack.time},
            dims=["time"],
        )
        daily_cc = cc_da.resample(time=cadence).max()
        daily_cc = daily_cc.reindex(time=full_steps, fill_value=np.nan)
        stack_resampled = stack_resampled.assign_coords(eo_cloud_cover=daily_cc)

    # Apply normalization if requested
    if normalize:
        stack_rechunk = stack_resampled.chunk({"y": -1, "x": -1})
        stack_resampled = xr.apply_ufunc(
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
    timestack = stack_resampled.loc[
        ..., y_utm + buffer : y_utm - buffer, x_utm - buffer : x_utm + buffer
    ]

    # Track percent of pixels that are NaN
    nan_counts = timestack.isnull().sum(dim=("band", "y", "x"))
    total = len(band_names) * tile_size * tile_size
    pct_nans = (nan_counts / total) * 100
    timestack = timestack.assign_coords(pct_nans=("time", pct_nans.values))

    # Compute cloud mask if requested
    if cloudmask is not False:
        if callable(cloudmask):
            cloud_mask_da = cloudmask(timestack)
        elif isinstance(cloudmask, str):
            cloud_mask_da = cloud_pix_mask(timestack, method=cloudmask)
        else:
            raise ValueError(
                f"cloudmask must be False, a method name string, or a callable. "
                f"Got: {type(cloudmask)}"
            )
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
        if cloudmask:
            pct_cloudy = pctcloudypix_inmask(timestack)
            timestack = timestack.assign_coords(
                pctcloudypix_inmask=("time", pct_cloudy.data)
            )

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
