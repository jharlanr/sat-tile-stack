"""
Unit tests for sat_tile_stack.append — generic data attachment API.

Tests append_band, append_timeseries, and append_metadata on synthetic data.
All tests use synthetic data and do NOT require internet access.
Run with: pytest tests/test_append.py -v

NOTE: There is a known issue with append_band when the source DataArray has
extra coordinates from STAC metadata (e.g., 'title'). These coordinates
are not present in the new band, causing xr.concat to fail. This is tracked
and will be fixed separately.
"""

import numpy as np
import pytest
import xarray as xr

from sat_tile_stack.append import append_band, append_timeseries, append_metadata


# ===========================================================================
# append_band — add spatial bands to a timestack
# ===========================================================================

class TestAppendBand:
    """Tests for appending spatial bands to an existing timestack."""

    def test_append_static_2d_numpy(self, synthetic_timestack):
        """A 2D numpy array should be broadcast across time and appended."""
        dem = np.ones((16, 16), dtype=np.float32) * 500.0
        result = append_band(synthetic_timestack, dem, "DEM")

        assert "DEM" in result.band.values, "DEM band should be present"
        assert len(result.band) == 4, "Should have 4 bands (3 original + DEM)"
        assert result.sel(band="DEM").shape == (7, 16, 16), "DEM should span all timesteps"
        # Verify the DEM value is consistent across time
        assert (result.sel(band="DEM") == 500.0).all(), "DEM should be constant across time"

    def test_append_temporal_3d_numpy(self, synthetic_timestack):
        """A 3D numpy array [time, y, x] should be appended directly."""
        ndvi = np.random.rand(7, 16, 16).astype(np.float32)
        result = append_band(synthetic_timestack, ndvi, "NDVI")

        assert "NDVI" in result.band.values
        assert result.shape == (7, 4, 16, 16), f"Expected (7,4,16,16), got {result.shape}"

    def test_append_xarray_dataarray(self, synthetic_timestack):
        """An xarray DataArray with matching coords should append cleanly."""
        ts = synthetic_timestack
        extra = xr.DataArray(
            np.ones((7, 16, 16), dtype=np.float32),
            dims=("time", "y", "x"),
            coords={"time": ts.time, "y": ts.y, "x": ts.x},
        )
        result = append_band(ts, extra, "custom_band")

        assert "custom_band" in result.band.values

    def test_append_preserves_original_bands(self, synthetic_timestack):
        """Original bands should be unchanged after appending a new one."""
        original_data = synthetic_timestack.sel(band="B04").values.copy()
        result = append_band(synthetic_timestack, np.zeros((16, 16)), "new")

        np.testing.assert_array_equal(
            result.sel(band="B04").values,
            original_data,
            err_msg="Original band data should not change"
        )

    def test_append_multiple_bands_sequentially(self, synthetic_timestack):
        """Should be able to append multiple bands one after another."""
        result = synthetic_timestack
        result = append_band(result, np.zeros((16, 16)), "band_A")
        result = append_band(result, np.ones((16, 16)), "band_B")

        assert len(result.band) == 5, "Should have 5 bands (3 + 2 appended)"
        assert "band_A" in result.band.values
        assert "band_B" in result.band.values

    def test_invalid_ndim_raises(self, synthetic_timestack):
        """4D array should raise ValueError (only 2D and 3D accepted)."""
        bad = np.ones((7, 3, 16, 16))
        with pytest.raises(ValueError, match="2D or 3D"):
            append_band(synthetic_timestack, bad, "bad")

    def test_1d_raises(self, synthetic_timestack):
        """1D array should raise ValueError."""
        bad = np.ones(16)
        with pytest.raises(ValueError, match="2D or 3D"):
            append_band(synthetic_timestack, bad, "bad")


# ===========================================================================
# append_timeseries — add 1D time-indexed coordinates
# ===========================================================================

class TestAppendTimeseries:
    """Tests for appending 1D time series as coordinates."""

    def test_basic_append(self, synthetic_timestack):
        """Should add a named coordinate on the time dimension."""
        series = np.arange(7, dtype=np.float32) * 10
        result = append_timeseries(synthetic_timestack, series, "temperature")

        assert "temperature" in result.coords, "Coordinate should be present"
        assert len(result.coords["temperature"]) == 7

    def test_values_preserved(self, synthetic_timestack):
        """Appended values should match the input exactly."""
        series = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7], dtype=np.float32)
        result = append_timeseries(synthetic_timestack, series, "water_area")

        np.testing.assert_array_almost_equal(
            result.coords["water_area"].values, series
        )

    def test_wrong_length_raises(self, synthetic_timestack):
        """Series with wrong length should raise ValueError."""
        short = np.arange(5)
        with pytest.raises(ValueError, match="length"):
            append_timeseries(synthetic_timestack, short, "bad")

    def test_append_multiple_timeseries(self, synthetic_timestack):
        """Should be able to append multiple time series."""
        result = synthetic_timestack
        result = append_timeseries(result, np.zeros(7), "series_a")
        result = append_timeseries(result, np.ones(7), "series_b")

        assert "series_a" in result.coords
        assert "series_b" in result.coords

    def test_accepts_list(self, synthetic_timestack):
        """Should accept a plain Python list as input."""
        result = append_timeseries(synthetic_timestack, [1, 2, 3, 4, 5, 6, 7], "counts")
        assert "counts" in result.coords


# ===========================================================================
# append_metadata — add scalar attributes
# ===========================================================================

class TestAppendMetadata:
    """Tests for attaching scalar metadata as attributes."""

    def test_string_value(self, synthetic_timestack):
        """Should store a string attribute."""
        result = append_metadata(synthetic_timestack, "lake_name", "Lake Hazen")
        assert result.attrs["lake_name"] == "Lake Hazen"

    def test_numeric_value(self, synthetic_timestack):
        """Should store a numeric attribute."""
        result = append_metadata(synthetic_timestack, "elevation", 884.4)
        assert result.attrs["elevation"] == 884.4

    def test_integer_value(self, synthetic_timestack):
        """Should store an integer attribute."""
        result = append_metadata(synthetic_timestack, "year", 2019)
        assert result.attrs["year"] == 2019

    def test_list_value(self, synthetic_timestack):
        """Should store a list attribute."""
        classes = ["ND", "ED", "LD", "CD"]
        result = append_metadata(synthetic_timestack, "classes", classes)
        assert result.attrs["classes"] == classes

    def test_overwrite_existing(self, synthetic_timestack):
        """Setting the same key twice should overwrite the first value."""
        result = append_metadata(synthetic_timestack, "version", "1.0")
        result = append_metadata(result, "version", "2.0")
        assert result.attrs["version"] == "2.0"

    def test_does_not_modify_data(self, synthetic_timestack):
        """Appending metadata should not change the data values."""
        original = synthetic_timestack.values.copy()
        result = append_metadata(synthetic_timestack, "key", "value")
        np.testing.assert_array_equal(result.values, original)
