"""
Shared fixtures for sat-tile-stack test suite.

Provides reusable test data, configurations, and STAC catalog connections
used across multiple test modules.

Usage:
    Fixtures are automatically available to all test files in this directory.
    No need to import — pytest discovers them from conftest.py.

    # In any test file:
    def test_something(synthetic_timestack):
        assert synthetic_timestack.shape == (7, 3, 16, 16)
"""

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# Greenland supraglacial lake — used for integration tests
TEST_CENTROID = (-49.4957449, 68.72493445)
TEST_TIME_RANGE = "2019-06-08/2019-06-14"
TEST_TILE_SIZE = 64
TEST_PIX_RES = 10


# ---------------------------------------------------------------------------
# Synthetic data fixtures (no internet required)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_timestack():
    """
    Minimal synthetic timestack for unit testing.

    Returns a [7, 3, 16, 16] DataArray with random float32 values
    simulating 7 daily timesteps of RGB imagery at 16x16 pixels.
    Band names are Sentinel-2 style: B04, B03, B02.
    """
    n_time, n_bands, size = 7, 3, 16
    data = np.random.rand(n_time, n_bands, size, size).astype(np.float32) * 3000
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(n_time),
            "band": ["B04", "B03", "B02"],
            "y": np.arange(size, dtype=float),
            "x": np.arange(size, dtype=float),
        },
    )


@pytest.fixture
def synthetic_timestack_with_swir():
    """
    Synthetic timestack including B11 (SWIR1) band for cloud mask testing.

    Returns a [7, 4, 16, 16] DataArray with bands: B04, B03, B02, B11.
    B11 values are set to a range that spans the Williamson threshold (1400).
    """
    n_time, size = 7, 16
    bands = ["B04", "B03", "B02", "B11"]
    data = np.random.rand(n_time, len(bands), size, size).astype(np.float32) * 3000
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(n_time),
            "band": bands,
            "y": np.arange(size, dtype=float),
            "x": np.arange(size, dtype=float),
        },
    )


@pytest.fixture
def synthetic_timestack_with_scl():
    """
    Synthetic timestack including SCL band for cloud mask testing.

    Returns a [7, 4, 16, 16] DataArray with bands: B04, B03, B02, SCL.
    SCL values default to 4 (vegetation = clear sky).
    """
    n_time, size = 7, 16
    bands = ["B04", "B03", "B02", "SCL"]
    data = np.random.rand(n_time, len(bands), size, size).astype(np.float32) * 3000
    # Set SCL to class 4 (vegetation = clear) by default
    data[:, 3, :, :] = 4.0
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(n_time),
            "band": bands,
            "y": np.arange(size, dtype=float),
            "x": np.arange(size, dtype=float),
        },
    )


@pytest.fixture
def synthetic_timestack_with_mask():
    """
    Synthetic timestack with a spatial mask band appended.

    Returns a [7, 4, 16, 16] DataArray with bands: B04, B03, B02, mask.
    The mask is a circle of 1s (inside) and 0s (outside) centered on the tile.
    """
    n_time, size = 7, 16
    bands = ["B04", "B03", "B02", "mask"]
    data = np.random.rand(n_time, len(bands), size, size).astype(np.float32) * 3000

    # Create circular mask
    yy, xx = np.mgrid[:size, :size]
    center = size // 2
    mask = ((yy - center)**2 + (xx - center)**2 < (size // 3)**2).astype(np.float32)
    data[:, 3, :, :] = mask

    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(n_time),
            "band": bands,
            "y": np.arange(size, dtype=float),
            "x": np.arange(size, dtype=float),
        },
    )


# ---------------------------------------------------------------------------
# Integration test fixtures (requires internet)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stac_catalog():
    """
    Planetary Computer STAC catalog client.

    Session-scoped so the connection is reused across all integration tests.
    Skips if pystac_client or planetary_computer are not installed.
    """
    pystac_client = pytest.importorskip("pystac_client")
    planetary_computer = pytest.importorskip("planetary_computer")

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# ---------------------------------------------------------------------------
# Custom markers registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require internet access (deselect with '-m \"not integration\"')"
    )
