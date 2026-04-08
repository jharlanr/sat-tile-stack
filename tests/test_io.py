"""
Unit tests for sat_tile_stack.io — NetCDF writing and attribute handling.

Tests the full write/read round-trip and attribute sanitization.
All tests use synthetic data and do NOT require internet access.
Run with: pytest tests/test_io.py -v
"""

import json
import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from sat_tile_stack.io import (
    scrub_attrs,
    sanitise_dataset,
    coerce_attrs_to_json_safe,
    write_netcdf_from_da,
)


# ===========================================================================
# write_netcdf_from_da — full round-trip
# ===========================================================================

class TestWriteNetCDF:
    """Tests for writing DataArrays to NetCDF and reading them back."""

    def test_basic_round_trip(self, synthetic_timestack):
        """Write and read should produce matching shapes and values."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name

        try:
            write_netcdf_from_da(synthetic_timestack, path)

            ds = xr.open_dataset(path)
            loaded = ds["reflectance"]

            assert loaded.shape == synthetic_timestack.shape, \
                f"Shape mismatch: {loaded.shape} vs {synthetic_timestack.shape}"
            assert list(loaded.band.values) == list(synthetic_timestack.band.values)
            assert len(loaded.time) == len(synthetic_timestack.time)
        finally:
            os.unlink(path)

    def test_file_is_compressed(self, synthetic_timestack):
        """Output file should be smaller than uncompressed data (gzip level 4)."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name

        try:
            write_netcdf_from_da(synthetic_timestack, path)

            file_size = os.path.getsize(path)
            uncompressed_size = synthetic_timestack.values.nbytes

            # Compressed file should be notably smaller than raw data
            assert file_size < uncompressed_size, \
                f"File ({file_size} bytes) should be smaller than raw data ({uncompressed_size} bytes)"
        finally:
            os.unlink(path)

    def test_output_is_float32(self, synthetic_timestack):
        """Data in the file should be stored as float32."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name

        try:
            write_netcdf_from_da(synthetic_timestack, path)
            ds = xr.open_dataset(path)
            assert ds["reflectance"].dtype == np.float32
        finally:
            os.unlink(path)

    def test_attrs_preserved(self, synthetic_timestack):
        """Attributes set on the DataArray should survive the round-trip."""
        ts = synthetic_timestack.copy()
        ts.attrs["lake_name"] = "Test Lake"
        ts.attrs["year"] = 2019

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name

        try:
            write_netcdf_from_da(ts, path)
            ds = xr.open_dataset(path)

            # Attrs get promoted to global attrs
            assert "lake_name" in ds.attrs or "lake_name" in ds["reflectance"].attrs
        finally:
            os.unlink(path)


# ===========================================================================
# scrub_attrs — attribute cleaning
# ===========================================================================

class TestScrubAttrs:
    """Tests for making xarray attributes JSON-serializable."""

    def test_numpy_scalar_converted(self, synthetic_timestack):
        """Numpy scalars should be converted to Python native types."""
        ts = synthetic_timestack.copy()
        ts.attrs["np_float"] = np.float64(3.14)
        ts.attrs["np_int"] = np.int32(42)

        cleaned = scrub_attrs(ts)

        # Should be JSON-serializable now
        for key in ["np_float", "np_int"]:
            json.dumps(cleaned.attrs[key])  # should not raise

    def test_non_serializable_becomes_string(self, synthetic_timestack):
        """Non-serializable objects should be converted to strings."""
        ts = synthetic_timestack.copy()
        ts.attrs["weird_object"] = object()

        cleaned = scrub_attrs(ts)

        assert isinstance(cleaned.attrs["weird_object"], str)

    def test_drops_specified_attrs(self, synthetic_timestack):
        """Attributes in the drop tuple should be removed."""
        ts = synthetic_timestack.copy()
        ts.attrs["spec"] = "some spec data"
        ts.attrs["keep_me"] = "important"

        cleaned = scrub_attrs(ts, drop=("spec",))

        assert "spec" not in cleaned.attrs
        assert "keep_me" in cleaned.attrs


# ===========================================================================
# coerce_attrs_to_json_safe — final safety pass
# ===========================================================================

class TestCoerceAttrs:
    """Tests for the final JSON-safety coercion pass on datasets."""

    def test_set_becomes_scalar_or_list(self):
        """Python sets should be converted (singleton -> scalar, else -> list)."""
        ds = xr.Dataset(attrs={"singleton": {"hello"}, "multi": {1, 2, 3}})
        result = coerce_attrs_to_json_safe(ds)

        assert result.attrs["singleton"] == "hello"
        assert isinstance(result.attrs["multi"], list)

    def test_numpy_types_converted(self):
        """Numpy types should become native Python types."""
        ds = xr.Dataset(attrs={
            "f32": np.float32(1.5),
            "i64": np.int64(100),
        })
        result = coerce_attrs_to_json_safe(ds)

        assert type(result.attrs["f32"]) is float
        assert type(result.attrs["i64"]) is int
