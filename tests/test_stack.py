"""
Integration tests for sat_tile_stack.stack — the core stacking pipeline.

These tests hit Microsoft Planetary Computer and require internet access.
They use a small tile (64x64) and short time range (1 week) for speed.

Run with:
    pytest tests/test_stack.py -v -m integration

Skip in CI or offline environments:
    pytest tests/ -v -m "not integration"
"""

import numpy as np
import pytest

from sat_tile_stack import sattile_stack

# Test parameters — Greenland supraglacial lake
CENTROID = (-49.4957449, 68.72493445)
TIME_RANGE = "2019-06-08/2019-06-14"
TILE_SIZE = 64
PIX_RES = 10


# ===========================================================================
# Basic stacking
# ===========================================================================

@pytest.mark.integration
class TestBasicStack:
    """Tests for core sattile_stack() functionality."""

    def test_rgb_shape_and_dims(self, stac_catalog):
        """Basic S2 RGB stack should have correct shape and dimensions."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, cloudmask=False, pull_to_mem=True,
        )

        assert stack.dims == ("time", "band", "y", "x")
        assert list(stack.band.values) == ["B04", "B03", "B02"]
        assert len(stack.time) == 7, f"Expected 7 daily steps, got {len(stack.time)}"
        assert stack.dtype == np.float32

    def test_has_pct_nans_coordinate(self, stac_catalog):
        """Stack should include a pct_nans coordinate on the time dimension."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, pull_to_mem=True,
        )

        assert "pct_nans" in stack.coords, "Missing pct_nans coordinate"

    def test_has_cloud_cover_coordinate(self, stac_catalog):
        """S2 stacks should include eo_cloud_cover metadata."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, pull_to_mem=True,
        )

        assert "eo_cloud_cover" in stack.coords, "Missing eo_cloud_cover coordinate"

    def test_empty_search_raises(self, stac_catalog):
        """Searching a location/time with no data should raise ValueError."""
        with pytest.raises(ValueError, match="No items found"):
            sattile_stack(
                stac_catalog, (0.0, 0.0), ["B04"],  # middle of the ocean
                collection="sentinel-2-l2a",
                time_range="1990-01-01/1990-01-02",  # before S2 existed
                pull_to_mem=True,
            )


# ===========================================================================
# Cloud masking
# ===========================================================================

@pytest.mark.integration
class TestCloudMask:
    """Tests for cloud masking options in sattile_stack()."""

    def test_scl_cloudmask(self, stac_catalog):
        """SCL cloud mask should produce a binary 'cloudmask' band."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02", "SCL"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, cloudmask="scl", pull_to_mem=True,
        )

        bands = list(stack.band.values)
        assert "cloudmask" in bands, f"cloudmask missing. Bands: {bands}"
        assert "SCL" in bands, "SCL band should still be present"

        # Verify binary
        cm = stack.sel(band="cloudmask").values
        unique = set(np.unique(cm[~np.isnan(cm)]))
        assert unique.issubset({0.0, 1.0}), f"Cloud mask not binary: {unique}"

    def test_williamson_cloudmask(self, stac_catalog):
        """Williamson cloud mask should produce a 'cloudmask' band."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02", "B11"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, cloudmask="williamson", pull_to_mem=True,
        )

        assert "cloudmask" in list(stack.band.values)

    def test_custom_cloudmask_callable(self, stac_catalog):
        """A custom callable cloud mask function should work."""
        # Trivial mask: everything is clear
        def all_clear(ts):
            import xarray as xr
            return xr.zeros_like(ts.isel(band=0).drop_vars("band")).astype("uint8")

        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, cloudmask=all_clear, pull_to_mem=True,
        )

        assert "cloudmask" in list(stack.band.values)
        cm = stack.sel(band="cloudmask").values
        assert (cm[~np.isnan(cm)] == 0).all(), "Custom mask should be all clear"


# ===========================================================================
# Temporal cadence and aggregation
# ===========================================================================

@pytest.mark.integration
class TestCadenceAndAggregation:
    """Tests for temporal resampling options."""

    def test_2day_cadence(self, stac_catalog):
        """2-day cadence over 7 days should produce 4 timesteps."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            cadence="2D", normalize=False, pull_to_mem=True,
        )

        assert len(stack.time) == 4, f"Expected 4 timesteps, got {len(stack.time)}"

    def test_nearest_aggregation(self, stac_catalog):
        """'nearest' aggregation should not blend scenes."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            cadence="2D", aggregation="nearest",
            normalize=False, pull_to_mem=True,
        )

        assert len(stack.time) == 4

    def test_weekly_cadence(self, stac_catalog):
        """Weekly cadence over 7 days should produce 1-2 timesteps."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            cadence="W", normalize=False, pull_to_mem=True,
        )

        assert 1 <= len(stack.time) <= 2


# ===========================================================================
# Normalization
# ===========================================================================

@pytest.mark.integration
class TestNormalization:
    """Tests for IQR-based robust normalization."""

    def test_normalized_values_in_0_1(self, stac_catalog):
        """Normalized pixel values should be in [0, 1]."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=True, pull_to_mem=True,
        )

        vals = stack.values[~np.isnan(stack.values)]
        assert vals.min() >= 0.0, f"Min value {vals.min()} < 0"
        assert vals.max() <= 1.0, f"Max value {vals.max()} > 1"

    def test_unnormalized_values_raw(self, stac_catalog):
        """Without normalization, S2 values should be raw reflectance (0–10000+)."""
        stack = sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=PIX_RES, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, pull_to_mem=True,
        )

        vals = stack.values[~np.isnan(stack.values)]
        assert vals.max() > 1.0, "Raw S2 values should be > 1 (scaled by 10000)"


# ===========================================================================
# Resolution warning
# ===========================================================================

@pytest.mark.integration
class TestResolutionWarning:
    """Tests for sub-native-resolution warnings."""

    def test_fine_resolution_prints_warning(self, stac_catalog, capsys):
        """Requesting pix_res < native should print a warning."""
        sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=3, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, pull_to_mem=True,
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out, "Should print a resolution warning"
        assert "finer than the native" in captured.out

    def test_native_resolution_no_warning(self, stac_catalog, capsys):
        """Requesting pix_res == native should NOT print a warning."""
        sattile_stack(
            stac_catalog, CENTROID, ["B04", "B03", "B02"],
            collection="sentinel-2-l2a",
            pix_res=10, tile_size=TILE_SIZE, time_range=TIME_RANGE,
            normalize=False, pull_to_mem=True,
        )

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
