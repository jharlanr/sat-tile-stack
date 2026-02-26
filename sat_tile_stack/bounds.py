"""
Geospatial bounds and masking utilities.

Provides functions for creating bounding boxes, spatial masks, and
computing statistics within masked regions.
"""

import math
import numpy as np
import pyproj
import xarray as xr
import rioxarray  # noqa: F401 — activates .rio accessor on xarray objects
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import transform


def sat_mask_array(array_in, gdf_geojson, feature_id=None):
    """
    Create a mask array from a GeoDataFrame.

    Parameters
    ----------
    array_in : xarray.DataArray
        Input data with shape [time, band, y, x] and CRS information.
    gdf_geojson : geopandas.GeoDataFrame
        GeoDataFrame containing geometries to rasterize as mask.
    feature_id : optional
        Not currently used, reserved for future feature selection.

    Returns
    -------
    mask_4d : xarray.DataArray
        4D mask array with shape [time, band=1, y, x] where 1.0 = inside
        geometry, 0.0 = outside.
    """
    # Align GeoDataFrame CRS to input array
    gdf = gdf_geojson.to_crs(array_in.rio.crs)

    # Rasterize to a 2D array of floats (1.0 inside, 0.0 outside)
    shapes = [(geom, 1.0) for geom in gdf.geometry]
    mask2d = rasterize(
        shapes,
        out_shape=(array_in.sizes["y"], array_in.sizes["x"]),
        transform=array_in.rio.transform(),
        fill=0.0,
        dtype="float32",
    )

    # Wrap as an xarray DataArray with dims ('y', 'x')
    mask = xr.DataArray(
        mask2d,
        dims=("y", "x"),
        coords={"y": array_in.y, "x": array_in.x},
        name="polygon_mask",
    )

    # Expand to time & band dims
    mask_4d = (
        mask.expand_dims(time=array_in.time)
        .expand_dims(band=["mask"], axis=1)
        .transpose("time", "band", "y", "x")
    )

    # Propagate all non-band coords from the input
    non_band = {
        nm: crd for nm, crd in array_in.coords.items() if "band" not in crd.dims
    }
    mask_4d = mask_4d.assign_coords(non_band)

    # For each original band-coord, give the mask one entry
    for nm, crd in array_in.coords.items():
        if crd.dims == ("band",):
            default = np.nan if np.issubdtype(crd.dtype, np.number) else "mask"
            one_val = xr.DataArray(
                [default],
                coords={"band": ["mask"]},
                dims=("band",),
                name=nm,
            )
            mask_4d = mask_4d.assign_coords({nm: one_val})

    return mask_4d


def bounds_latlon_around(center_lon, center_lat, side_m=10000):
    """
    Create a bounding box in lat/lon coordinates around a center point.

    Parameters
    ----------
    center_lon : float
        Longitude of center point in decimal degrees (EPSG:4326).
    center_lat : float
        Latitude of center point in decimal degrees (EPSG:4326).
    side_m : float, optional
        Length of box side in meters (default: 10000).

    Returns
    -------
    bounds : tuple
        (minx, miny, maxx, maxy) in lon/lat coordinates.
    """
    # Set up transformers
    centroid = (center_lon, center_lat)
    epsg_code = int(best_crs_for_point(*centroid).to_authority()[1])
    to_ps = pyproj.Transformer.from_crs(4326, epsg_code, always_xy=True).transform
    to_ll = pyproj.Transformer.from_crs(epsg_code, 4326, always_xy=True).transform

    # Project centroid to meters
    x0, y0 = to_ps(center_lon, center_lat)

    # Build square around centroid
    half = side_m / 2.0
    sq_m = box(x0 - half, y0 - half, x0 + half, y0 + half)

    # Reproject square back to lat/lon and grab bounds
    sq_ll = transform(to_ll, sq_m)
    return sq_ll.bounds


def best_crs_for_point(lon, lat):
    """
    Choose a projected CRS that minimizes distortion for a point.

    Parameters
    ----------
    lon : float
        Longitude in decimal degrees.
    lat : float
        Latitude in decimal degrees.

    Returns
    -------
    crs : pyproj.CRS
        Appropriate projected CRS:
        - |lat| >= 60: Polar Stereographic (EPSG:3413 North / 3031 South)
        - else: UTM zone based on longitude
    """
    if lat >= 60:
        # Arctic Polar Stereographic
        return pyproj.CRS.from_epsg(3413)
    elif lat <= -60:
        # Antarctic Polar Stereographic
        return pyproj.CRS.from_epsg(3031)
    else:
        # UTM
        zone_number = int(math.floor((lon + 180) / 6) + 1)
        is_south = lat < 0
        proj4 = (
            f"+proj=utm +zone={zone_number} "
            f"+{'south' if is_south else 'north'} +datum=WGS84 +units=m +no_defs"
        )
        return pyproj.CRS.from_proj4(proj4)


def pctnanpix_inmask(timestack):
    """
    Compute percentage of NaN pixels within a mask for each timestep.

    Parameters
    ----------
    timestack : xarray.DataArray
        DataArray with dimensions [time, band, y, x], must include "mask"
        and RGB bands (B04, B03, B02).

    Returns
    -------
    pct_nan : xarray.DataArray
        Percentage (0-1) of pixels inside mask that are NaN for each timestep.
    """
    lakemask_bool = timestack.sel(band="mask") == 1  # shape [time, y, x]
    total_pixels = lakemask_bool.sum(dim=["y", "x"])  # shape [time]
    rgb_stack = timestack.sel(band=["B04", "B03", "B02"])
    nan_mask = np.isnan(rgb_stack).all(dim="band")
    nan_in_mask = nan_mask & lakemask_bool  # shape [time, y, x]
    nan_count = nan_in_mask.sum(dim=["y", "x"])
    pct_nan = nan_count / total_pixels

    return pct_nan


def pctcloudypix_inmask(timestack):
    """
    Compute percentage of cloudy pixels within a mask for each timestep.

    Parameters
    ----------
    timestack : xarray.DataArray
        DataArray with dimensions [time, band, y, x], must include "mask"
        and "cloudmask" bands.

    Returns
    -------
    pct_cloudy : xarray.DataArray
        Percentage (0-1) of pixels inside mask that are cloudy for each timestep.
    """
    lakemask_bool = timestack.sel(band="mask") == 1  # shape [time, y, x]
    cloudmask_bool = timestack.sel(band="cloudmask") == 1  # shape [time, y, x]
    total_pixels = lakemask_bool.sum(dim=["y", "x"])  # shape [time]
    cloudy_in_mask = cloudmask_bool & lakemask_bool  # shape [time, y, x]
    cloudy_count = cloudy_in_mask.sum(dim=["y", "x"])
    pct_cloudy = cloudy_count / total_pixels

    return pct_cloudy
