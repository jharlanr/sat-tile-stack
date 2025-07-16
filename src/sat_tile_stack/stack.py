## ======= ##
## IMPORTS ##
## ======= ##

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import os
import dask
import dask.array
import math

import datetime

from collections import Counter

import pystac_client
from pystac.extensions.projection import ProjectionExtension as proj

import planetary_computer
import rasterio
import rasterio.features
from rasterio.features import rasterize

import stackstac
import pyproj

import dask.diagnostics

from shapely.geometry import box
from shapely.ops import transform

from scipy.ndimage import binary_propagation
from scipy.ndimage import label

# imports from my package
from .bounds import sat_mask_array, bounds_latlon_around, best_crs_for_point, pctnanpix_inmask, pctcloudypix_inmask
from .utils  import combo_scaler, cloud_pix_mask


## ========= ##
## FUNCTIONS ##
## ========= ##

## FUNCTION FOR GENERATE A STACK FOR A GIVEN LOCATION ACROSS A DATE RANGE, EACH DAY
def sattile_stack(catalog,
                  centroid,
                  band_names,
                  pix_res=10,
                  tile_size=1024,
                  time_range='2019-05-01/2019-09-30',
                  normalize=True,
                  cloudmask=False,
                  mask=None,
                  pull_to_mem=False):
    """
    Generate a daily, multi-band image time‐stack centered on a point.

    This function:
      1. Searches a STAC catalog (Sentinel-2 L2A) for all scenes covering
         a square of size (in meters) around `centroid`
         between the dates in `time_range`.
      2. Builds a 4D xarray.DataArray (time, band, y, x) using stackstac.
      3. Resamples to a daily cadence, carrying forward the maximum
         cloud‐cover metadata for each day.
      4. Optionally applies a robust per‐band normalization (to [0, 1]).
      5. Crops to a square “tile” of `tile_size × tile_size` pixels around
         the centroid.
      6. Optionally computes into memory (with a dask progress bar).

    Parameters
    ----------
    centroid : tuple of float
        (longitude, latitude) in decimal degrees of the point of interest.
    band_names : list of str
        Sentinel-2 band names (e.g. ['B04','B03','B02','B08','B11']).
    tile_size : int, optional
        Size of the output tile in pixels (square) (default: 512).
    time_range : str, optional
        ISO8601 date range “YYYY-MM-DD/YYYY-MM-DD” for imagery search
        (default: '2019-05-01/2019-09-30').
    normalize : bool, optional
        If True, apply a robust (median/IQR) normalization to each band
        so values are scaled into [0,1] (default: True).
    mask: string, optional
        If None, do not generate a mask, if not None (e.g., filepath to a .geojson),
        generate mask and append to DataArray
    pull_to_mem : bool, optional
        If True, triggers `timestack.compute()` and returns an in-memory
        xarray.DataArray; otherwise returns a lazy dask-backed DataArray
        (default: False).

    Returns
    -------
    xarray.DataArray
        DataArray of shape (time, band, tile_size, tile_size). Coordinates
        are:
          - time: daily steps over `time_range`
          - band: the requested `band_names`
          - y, x: UTM coordinates of the tile
          - eo_cloud_cover: cloud cover percentage
          - pct_nans: percent of tile that is nans

    Notes
    -----
    - Requires `stackstac`, `xarray`, `dask`, and a STAC `catalog` in scope.
    - CRS for reprojection is taken from the first item in the search.
    """
    
    # SEARCH IMAGERY CATALOG FOR ITEMS MATCHING LOCATION AND DATE RANGE
    bounds_latlon = bounds_latlon_around(*centroid, side_m=pix_res*tile_size*1.1)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds_latlon,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 100}} )
    items = search.item_collection()
    
    # CREATE THE STACK
    stack = stackstac.stack(items, 
                            epsg=proj.ext(items[0]).epsg, 
                            assets=band_names,
                            bounds_latlon=bounds_latlon, 
                            resolution=10,
                            chunksize=(1, 1, 4096, 4096),
                            fill_value=np.nan,
                            )
    
    # MASK OUT BLACK 0.0 PIXELS AS NaNs
    nodata_mask = (stack == 0).all(dim="band")   # True where every band is zero
    stack = stack.where(~nodata_mask)            # nodata → NaN, real zeros kept
    
    # SAMPLE DAILY, PRESERVING CLOUD COVER IN THE METADATA
    stack_daily = stack.resample(time="D").mean("time", keep_attrs=True)
    cc_da = xr.DataArray(
        np.array([item.properties["eo:cloud_cover"] for item in items]),
        coords={"time": stack.time},
        dims=["time"])
    start, end = time_range.split("/")
    full_days = pd.date_range(start, end, freq="D")
    stack_daily = stack_daily.reindex(time=full_days, fill_value=np.nan)
    daily_cc = cc_da.resample(time="D").max()
    daily_cc = daily_cc.reindex(time=full_days, fill_value=np.nan)
    stack_daily = stack_daily.assign_coords(eo_cloud_cover=daily_cc)
    # print(f"stack_daily shape: {stack_daily.shape}")
    
    
    # IF NORMALIZING
    if normalize:
        stack_rechunk = stack_daily.chunk({'y': -1, 'x': -1})
        stack_daily = xr.apply_ufunc(
            combo_scaler,            
            stack_rechunk,                   
            input_core_dims=[['y','x']],
            output_core_dims=[['y','x']],
            vectorize=True,       
            dask='parallelized',
            output_dtypes=[float],
            kwargs={'range_max':1},
            keep_attrs=True)

    # CROP TILES TO DESIRED SIZE
    pix_res = 10 #[m/pix]
    buffer = tile_size * pix_res/2 #[m]
    x_utm, y_utm = pyproj.Proj(stack.crs)(*centroid)
    timestack = stack_daily.loc[..., y_utm+buffer:y_utm-buffer, x_utm-buffer:x_utm+buffer]
    
    # TRACK PERCENT OF PIXELS THAT ARE NAN AND STORE AS NEW COORDINATE IN DATAARRAY
    nan_counts = timestack.isnull().sum(dim=('band', 'y', 'x'))
    total = len(band_names) * tile_size * tile_size
    pct_nans = (nan_counts / total) * 100
    timestack = timestack.assign_coords(pct_nans=('time', pct_nans.values))
    
    # IF DESIRED, COMPUTE A CLOUD MASK AND APPEND TO DATAARRAY
    if cloudmask is True:
        cloudmask = cloud_pix_mask(timestack, mask_method="Williamson2018b")  # [time, y, x]
        # Add a band dimension to cloud_mask
        cloudmask = cloudmask.expand_dims(dim={"band": ["cloudmask"]})  # [time, band=1, y, x]
        # Align coords and concatenate along band dimension
        timestack = xr.concat([timestack, cloudmask], dim="band")
    else:
        print(f"no cloudmask generation called")
    
    # IF THERE'S A REGION MASK, GENERATE AND APPEND MASK TO DATAARRAY
    if mask is not None:
        satmask = sat_mask_array(timestack, mask, feature_id=None)
        timestack = xr.concat([timestack, satmask], dim="band")
    else:
        print(f"no mask generation called")
        
    # TRACK PERCENT OF PIXELS WITHIN MASK THAT ARE NANS AND CLOUDY
    if mask is not None:
        # nans:  (note that this will place a 1 in the time vector if there is 100% NaNs within the lake area)
        pct_nan = pctnanpix_inmask(timestack)
        timestack = timestack.assign_coords(pctnanpix_inmask=("time", pct_nan.data))
        # cloudiness: (note that this will place a 1 in the time vector if there is 100% cloudiness within the lake area)
        # NOTE: this currently also places a 1 in the time vector if the image has NaNs inside the lake area
        pct_cloudy = pctcloudypix_inmask(timestack)
        timestack = timestack.assign_coords(pctcloudypix_inmask=("time", pct_cloudy.data))
    else:
        print(f"unable to produce pct_nans_inmask, pct_cloudy_inmask mask generation called")
        
    # CONVERT TO FLOAT32
    timestack = timestack.astype('float32')
    
    # PULL STACK INTO MEMORY
    if pull_to_mem:
        stack_to_pull = timestack
        print(f"pulling stack into memory, shape will be: {timestack.shape}")
        with dask.diagnostics.ProgressBar():
            timestack_mem = stack_to_pull.compute()
        print(f"shape of stack in memory: {timestack_mem.shape}")
        return timestack_mem
    else:
        print(f"timestack (not in mem) shape: {timestack.shape}\n")
        return timestack
    
    

# ## ========= ##
# ## RUN BLOCK ##
# ## ========= ##
# if __name__=="__main__":
    
#     ## CONNECT TO MICROSOFT PLANETARY COMPUTER
#     catalog = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1",
#         modifier=planetary_computer.sign_inplace,)
#     print(f"connected to Microsoft Planetary Computer")

#     # DECIDE WHETHER TO NORMALIZE THE IMAGERY UPON COMPLIING OR NOT
#     normalize = False

#     # BOUNDING BOX AROUND LAKE CENTROID
#     centroid = (-122.45, 37.83)
#     # centroid = (-122.45957, 37.80539) # CHRISSY FIELD MARSH
#     # centroid = (-49.495, 68.725) # NORTH LAKE

#     # SPECIFY TIME RANGE
#     time_range = '2019-05-01/2019-09-30'

#     # SPECIFY IMAGERY BANDS
#     band_names = ["B04",  # red (665 nm)
#                   "B03",  # green (560 nm)
#                   "B02",  # blue (490 nm)
#                   "B08",  # NIR (842 nm)
#                   "B11"]  # SWIR1 (1610 nm)

#     # CALL FUNCTION TO GENERATE TIMESTACK
#     timestack = sattile_stack(catalog, centroid, band_names, pix_res=10, tile_size=1024, time_range='2019-05-01/2019-09-30', normalize=True, pull_to_mem=True)
    
