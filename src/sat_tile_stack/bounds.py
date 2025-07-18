import pyproj
from shapely.geometry import box
from shapely.ops import transform
import math 
import geopandas as gpd
import xarray as xr
from rasterio.features import rasterize
import numpy as np

## FUNCTION TO CREATE A MASK USING A .geojson FILE
def sat_mask_array(array_in, gdf_geojson, feature_id=None):
    
    # align .geojson crs
    gdf = gdf_geojson.to_crs(array_in.rio.crs)

    # rasterize to a 2D array of floats (1.0 inside, 0.0 outside)
    shapes = [(geom, 1.0) for geom in gdf.geometry]
    mask2d = rasterize(
        shapes,
        out_shape=(array_in.sizes["y"], array_in.sizes["x"]),
        transform=array_in.rio.transform(),
        fill=0.0,
        dtype="float32")

    # wrap as an xarray DataArray, with dims ('y','x')
    mask = xr.DataArray(
        mask2d,
        dims=("y", "x"),
        coords={"y": array_in.y, "x": array_in.x},
        name="polygon_mask",
    )
    
    # expand to time & band dims
    mask_4d = (
        mask
        .expand_dims(time=array_in.time)           # dims: (time, y, x)
        .expand_dims(band=["mask"], axis=1)     # dims: (time, band, y, x)
        .transpose("time", "band", "y", "x")
    )
    
    # propagate all non‐band coords from the input
    non_band = {
        nm: crd for nm, crd in array_in.coords.items()
        if "band" not in crd.dims
    }
    mask_4d = mask_4d.assign_coords(non_band)
    
    # for each of the original band‐coords (like gsd, title, etc.), give the mask one entry
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

## FUNCTION TO DEFINE BOUNDING BOX AROUND A GIVEN CENTROID
def bounds_latlon_around(center_lon, center_lat, side_m=10000):
    """
    center_lon, center_lat : centroid in decimal degrees (EPSG:4326)
    side_m                 : length of box side in meters (default 10 km)
    returns                 : (minx, miny, maxx, maxy) in lon/lat
    """
    
    # SET UP TRANSFORMERS
    centroid = (center_lon, center_lat)
    epsg_code = int(best_crs_for_point(*centroid).to_authority()[1])
    print(f"epsg for bounds_latlon: {epsg_code}")
    to_ps = pyproj.Transformer.from_crs(4326, epsg_code, always_xy=True).transform
    to_ll = pyproj.Transformer.from_crs(epsg_code, 4326, always_xy=True).transform

    # PROJECT CENTROID TO METERS
    x0, y0 = to_ps(center_lon, center_lat)

    # BUILD SQUARE AROUND CENTROID
    half = side_m / 2.0
    sq_m = box(x0 - half, y0 - half, x0 + half, y0 + half)

    # REPROJECT SQUARE BACK TO LAT/LON AND GRAB BOUNDS
    sq_ll = transform(to_ll, sq_m)
    return sq_ll.bounds

## FUNCTION TO FIND BEST CRS FOR CENTROID
def best_crs_for_point(lon, lat):
    """
    Choose a projected CRS (EPSG) that minimizes distortion
    for a small box around (lon, lat).

    - |lat| ≥ 60° → Polar Stereographic (EPSG:3413 North / 3031 South)
    - else         → UTM zone based on lon

    Returns a pyproj.CRS object.
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
        is_south   = lat < 0
        # Construct a PROJ string for UTM:
        proj4 = (
            f"+proj=utm +zone={zone_number} "
            f"+{'south' if is_south else 'north'} +datum=WGS84 +units=m +no_defs"
        )
        return pyproj.CRS.from_proj4(proj4)
    
    
# FUNCTION TO COMPUTE PERCENTAGE OF NANS THAT ARE WITHIN A MASK FOR A GIVEN TILE
def pctnanpix_inmask(timestack):
    """
    Computes the percentage of clouded and clear pixels over the region of interest (e.g., lake)

    Parameters
    ----------
    timestack: (xarray.DataArray) DataArray with dimensions [time, band, y, x], including "mask" and RGB bands

    Returns
    -------
    pct_nan : (float) percent of pixels inside region (e.g., lake) that are NaNs
    """
    lakemask_bool = timestack.sel(band="mask")==1 # shape [time, y, x]
    total_pixels = lakemask_bool.sum(dim=["y", "x"]) # shape [time]
    rgb_stack = timestack.sel(band=["B04","B03","B02"])
    nan_mask = np.isnan(rgb_stack).all(dim="band")
    nan_in_mask = nan_mask & lakemask_bool # shape [time, y, x]
    nan_count = nan_in_mask.sum(dim=["y","x"])
    pct_nan = nan_count/total_pixels

    return pct_nan

# FUNCTION TO COMPUTE PERCENTAGE OF CLOUDY PIXELS THAT ARE WITHIN A MASK FOR A GIVEN TILE
def pctcloudypix_inmask(timestack):
    """
    Computes the percentage of clouded and clear pixels over the region of interest (e.g., lake)
    Uses a given pixel-wise cloudmask, which can be generated in multiple ways (e.g., see utils.py)

    Parameters
    ----------
    timestack : (xarray.DataArray) DataArray with dimensions [time, band, y, x], including "mask" and RGB bands

    Returns
    -------
    pct_cloudy : (float) percent of pixels inside region (e.g., lake) that are cloud-covered
    """
    lakemask_bool = timestack.sel(band="mask")==1 # shape [time, y, x], 1s are inside the lake mask (region of interest)
    cloudmask_bool = timestack.sel(band="cloudmask")==1 # shape: [time, y, x] (1s are cloudy pixels)
    total_pixels = lakemask_bool.sum(dim=["y", "x"]) # shape [time]
    cloudy_in_mask = cloudmask_bool & lakemask_bool # shape [time, y, x]
    cloudy_count = cloudy_in_mask.sum(dim=["y","x"])
    pct_cloudy = cloudy_count/total_pixels
    
    return pct_cloudy


