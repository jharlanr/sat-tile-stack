# src/sat_tile_stack/__init__.py

__version__="0.1.0"

from .stack import sattile_stack
from .bounds import sat_mask_array, bounds_latlon_around, best_crs_for_point, pctnanpix_inmask, pctcloudypix_inmask
from .io import scrub_attrs, sanitise_dataset, coerce_attrs_to_json_safe, write_netcdf_from_da, export_geotiff
from .visualize import timestack_to_movie, export_frame, multi_panel_frame, batch_movies
from .utils import combo_scaler, cloud_pix_mask
from .append import append_band, append_timeseries, append_metadata
from .ids import FeatureTracker

__all__ = [
    "sattile_stack",
    "sat_mask_array", "bounds_latlon_around", "best_crs_for_point", "pctnanpix_inmask", "pctcloudypix_inmask",
    "scrub_attrs", "sanitise_dataset", "coerce_attrs_to_json_safe", "write_netcdf_from_da", "export_geotiff",
    "timestack_to_movie", "export_frame", "multi_panel_frame", "batch_movies",
    "combo_scaler", "cloud_pix_mask",
    "append_band", "append_timeseries", "append_metadata",
    "FeatureTracker",
]

