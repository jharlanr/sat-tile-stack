# src/sat_tile_stack/__init__.py

__version__="0.1.0"

from .stack import sattile_stack
from .bounds import sat_mask_array, bounds_latlon_around, best_crs_for_point, pctnanpix_inmask, pctcloudypix_inmask
from .io import scrub_attrs, sanitise_dataset, coerce_attrs_to_json_safe, write_netcdf_from_da, write_netcdf, finalize_cf, export_geotiff
from .visualize import timestack_to_movie, export_frame, multi_panel_frame, batch_movies
from .utils import combo_scaler, cloud_pix_mask, SCL_CLOUDY_CLASSES
from .coregister import (
    add_raster_zarr, add_ndwi_mask, add_ndwi_from_stack,
    add_static_polygon, add_scalar_series,
)
# NB: import the batch fn under an alias — a bare `cf_check` here would
# shadow the `sat_tile_stack.cf_check` submodule (same name) and break
# `import sat_tile_stack.cf_check`.
from .cf_check import quick_cf_audit, check_file
from .cf_check import cf_check as cf_check_paths
from .append import append_band, append_timeseries, append_metadata
from .ids import FeatureTracker

__all__ = [
    "sattile_stack",
    "sat_mask_array", "bounds_latlon_around", "best_crs_for_point", "pctnanpix_inmask", "pctcloudypix_inmask",
    "scrub_attrs", "sanitise_dataset", "coerce_attrs_to_json_safe", "write_netcdf_from_da", "write_netcdf", "finalize_cf", "export_geotiff",
    "timestack_to_movie", "export_frame", "multi_panel_frame", "batch_movies",
    "combo_scaler", "cloud_pix_mask", "SCL_CLOUDY_CLASSES",
    "add_raster_zarr", "add_ndwi_mask", "add_ndwi_from_stack",
    "add_static_polygon", "add_scalar_series",
    "quick_cf_audit", "check_file", "cf_check_paths",
    "append_band", "append_timeseries", "append_metadata",
    "FeatureTracker",
]

