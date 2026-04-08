"""
Unit tests for sat_tile_stack.bounds — geospatial utilities.

Tests CRS selection, bounding box generation, and mask statistics.
All tests use synthetic data and do NOT require internet access.
Run with: pytest tests/test_bounds.py -v
"""

import numpy as np
import pytest

from sat_tile_stack.bounds import best_crs_for_point, pctnanpix_inmask, pctcloudypix_inmask


# ===========================================================================
# best_crs_for_point — optimal CRS selection
# ===========================================================================

class TestBestCRS:
    """Tests for automatic CRS selection based on geographic location."""

    def test_arctic_returns_polar_stereographic(self):
        """Locations above 60°N should use EPSG:3413 (Arctic Polar Stereo)."""
        crs = best_crs_for_point(-49.5, 68.7)  # Greenland
        assert crs.to_epsg() == 3413

    def test_antarctic_returns_polar_stereographic(self):
        """Locations below 60°S should use EPSG:3031 (Antarctic Polar Stereo)."""
        crs = best_crs_for_point(0.0, -75.0)  # Antarctica
        assert crs.to_epsg() == 3031

    def test_midlat_north_returns_utm(self):
        """Mid-latitude northern hemisphere should return a UTM north zone."""
        crs = best_crs_for_point(-73.9, 40.7)  # New York City
        epsg = crs.to_epsg()
        assert 32600 < epsg < 32661, f"Expected UTM north zone, got EPSG:{epsg}"

    def test_midlat_south_returns_utm(self):
        """Mid-latitude southern hemisphere should return a UTM south zone."""
        crs = best_crs_for_point(151.2, -33.9)  # Sydney
        epsg = crs.to_epsg()
        assert 32700 < epsg < 32761, f"Expected UTM south zone, got EPSG:{epsg}"

    def test_equator_returns_utm(self):
        """Equatorial location should return a UTM zone (not polar)."""
        crs = best_crs_for_point(-60.0, 0.0)  # Amazon
        epsg = crs.to_epsg()
        assert 32600 < epsg < 32661 or 32700 < epsg < 32761

    def test_exactly_60_north(self):
        """Latitude exactly at 60°N should use polar stereographic."""
        crs = best_crs_for_point(0.0, 60.0)
        assert crs.to_epsg() == 3413

    def test_just_below_60_north(self):
        """Latitude just below 60°N should use UTM."""
        crs = best_crs_for_point(0.0, 59.9)
        epsg = crs.to_epsg()
        assert epsg != 3413, "Should be UTM, not polar stereo"


# ===========================================================================
# pctnanpix_inmask — NaN pixel percentage within mask
# ===========================================================================

class TestPctNanInMask:
    """Tests for computing NaN pixel percentage within a spatial mask."""

    def test_no_nans_returns_zero(self, synthetic_timestack_with_mask):
        """If no pixels are NaN, percentage should be 0 for all timesteps."""
        pct = pctnanpix_inmask(synthetic_timestack_with_mask)
        assert (pct == 0.0).all(), f"Expected all zeros, got: {pct.values}"

    def test_all_nans_returns_one(self, synthetic_timestack_with_mask):
        """If all imagery pixels are NaN, percentage should be 1.0."""
        ts = synthetic_timestack_with_mask.copy()
        # Set all imagery bands (not mask) to NaN
        for band in ["B04", "B03", "B02"]:
            ts.loc[dict(band=band)] = np.nan

        pct = pctnanpix_inmask(ts)
        assert (pct == 1.0).all(), f"Expected all ones, got: {pct.values}"

    def test_partial_nans(self, synthetic_timestack_with_mask):
        """Partial NaN coverage should give a value between 0 and 1."""
        ts = synthetic_timestack_with_mask.copy()
        # Set NaN in the top-left quadrant only (all bands)
        for band in ["B04", "B03", "B02"]:
            ts.loc[dict(band=band, y=slice(0, 7), x=slice(0, 7))] = np.nan

        pct = pctnanpix_inmask(ts)
        assert (pct >= 0.0).all() and (pct <= 1.0).all()

    def test_custom_check_bands(self, synthetic_timestack_with_mask):
        """Should work with user-specified band names (not just B04/B03/B02)."""
        # Rename bands to non-S2 names
        ts = synthetic_timestack_with_mask.copy()
        ts = ts.assign_coords(band=["VV", "VH", "HH", "mask"])

        pct = pctnanpix_inmask(ts, check_bands=["VV", "VH"])
        assert (pct == 0.0).all()

    def test_default_check_bands_excludes_mask(self, synthetic_timestack_with_mask):
        """Default check_bands should auto-exclude 'mask' and 'cloudmask'."""
        pct = pctnanpix_inmask(synthetic_timestack_with_mask)
        # Should work without specifying check_bands
        assert pct.shape == (7,)


# ===========================================================================
# pctcloudypix_inmask — cloudy pixel percentage within mask
# ===========================================================================

class TestPctCloudyInMask:
    """Tests for computing cloudy pixel percentage within a spatial mask."""

    def test_no_clouds_returns_zero(self, synthetic_timestack_with_mask):
        """If cloudmask is all 0, percentage should be 0."""
        ts = synthetic_timestack_with_mask.copy()
        # Add a cloudmask band of all zeros (clear)
        import xarray as xr
        cm = xr.zeros_like(ts.sel(band="mask")).expand_dims(band=["cloudmask"])
        ts = xr.concat([ts, cm], dim="band")

        pct = pctcloudypix_inmask(ts)
        assert (pct == 0.0).all()

    def test_all_cloudy_returns_one(self, synthetic_timestack_with_mask):
        """If cloudmask is all 1, percentage should be 1.0."""
        import xarray as xr
        ts = synthetic_timestack_with_mask.copy()
        cm = xr.ones_like(ts.sel(band="mask")).expand_dims(band=["cloudmask"])
        ts = xr.concat([ts, cm], dim="band")

        pct = pctcloudypix_inmask(ts)
        assert (pct == 1.0).all()
