"""
Unit tests for sat_tile_stack.utils — cloud masking and normalization.

All tests use synthetic data and do NOT require internet access.
Run with: pytest tests/test_utils.py -v
"""

import numpy as np
import pytest

from sat_tile_stack.utils import combo_scaler, cloud_pix_mask


# ===========================================================================
# combo_scaler — IQR-based robust normalization
# ===========================================================================

class TestComboScaler:
    """Tests for the IQR-based normalization function."""

    def test_output_in_0_1_range(self):
        """Scaled values should be in [0, 1] by default."""
        x = np.random.rand(100) * 10000
        scaled = combo_scaler(x)
        assert np.nanmin(scaled) >= 0.0
        assert np.nanmax(scaled) <= 1.0

    def test_preserves_nans(self):
        """NaN values in input should remain NaN after scaling."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        scaled = combo_scaler(x)
        assert np.isnan(scaled[2]), "NaN at index 2 should be preserved"
        assert not np.isnan(scaled[0]), "Non-NaN values should not become NaN"

    def test_custom_range_max(self):
        """Output should respect the range_max parameter."""
        x = np.random.rand(50) * 100
        scaled = combo_scaler(x, range_max=255)
        assert np.nanmax(scaled) <= 255.0

    def test_constant_input(self):
        """Constant input (IQR=0) should not raise errors."""
        x = np.full(50, 42.0)
        # IQR is 0, so division by zero could occur — should handle gracefully
        scaled = combo_scaler(x)
        assert not np.any(np.isinf(scaled)), "Should not produce infinities"

    def test_single_value(self):
        """Single-element input should not crash."""
        x = np.array([5.0])
        scaled = combo_scaler(x)
        assert scaled.shape == (1,)

    def test_all_nans(self):
        """All-NaN input should return all NaN."""
        x = np.full(10, np.nan)
        scaled = combo_scaler(x)
        assert np.all(np.isnan(scaled))


# ===========================================================================
# cloud_pix_mask — pixel-wise cloud detection
# ===========================================================================

class TestCloudPixMaskWilliamson:
    """Tests for the Williamson2018b SWIR-threshold cloud masking method."""

    def test_all_cloudy(self, synthetic_timestack_with_swir):
        """B11 values above threshold (1400) should be flagged as cloudy."""
        ts = synthetic_timestack_with_swir.copy()
        ts.loc[dict(band="B11")] = 2000.0  # above 0.140 * 10000

        mask = cloud_pix_mask(ts, method="williamson")

        assert mask.shape == (7, 16, 16), f"Wrong shape: {mask.shape}"
        assert (mask == 1).all(), "All pixels should be cloudy"

    def test_all_clear(self, synthetic_timestack_with_swir):
        """B11 values below threshold should be flagged as clear."""
        ts = synthetic_timestack_with_swir.copy()
        ts.loc[dict(band="B11")] = 100.0  # well below threshold

        mask = cloud_pix_mask(ts, method="williamson")

        assert (mask == 0).all(), "All pixels should be clear"

    def test_nan_pixels_flagged_as_cloudy(self, synthetic_timestack_with_swir):
        """NaN pixels in B11 should be flagged as cloudy."""
        ts = synthetic_timestack_with_swir.copy()
        ts.loc[dict(band="B11")] = np.nan

        mask = cloud_pix_mask(ts, method="williamson")

        assert (mask == 1).all(), "NaN pixels should be marked as cloudy"

    def test_mixed_values(self, synthetic_timestack_with_swir):
        """Mix of cloudy and clear pixels should produce both 0s and 1s."""
        ts = synthetic_timestack_with_swir.copy()
        ts.loc[dict(band="B11", y=slice(0, 7))] = 2000.0   # cloudy top half
        ts.loc[dict(band="B11", y=slice(8, 15))] = 100.0    # clear bottom half

        mask = cloud_pix_mask(ts, method="williamson")

        assert 0 in mask.values, "Should have clear pixels"
        assert 1 in mask.values, "Should have cloudy pixels"

    def test_output_dtype(self, synthetic_timestack_with_swir):
        """Cloud mask should be uint8."""
        mask = cloud_pix_mask(synthetic_timestack_with_swir, method="williamson")
        assert mask.dtype == np.uint8

    def test_missing_b11_raises(self, synthetic_timestack):
        """Should raise ValueError if B11 band is not present."""
        with pytest.raises(ValueError, match="B11"):
            cloud_pix_mask(synthetic_timestack, method="williamson")


class TestCloudPixMaskSCL:
    """Tests for the SCL (Scene Classification Layer) cloud masking method."""

    def test_cloud_high_probability(self, synthetic_timestack_with_scl):
        """SCL class 9 (cloud high probability) should be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 9.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 1).all(), "Class 9 should be cloudy"

    def test_cloud_medium_probability(self, synthetic_timestack_with_scl):
        """SCL class 8 (cloud medium probability) should be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 8.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 1).all(), "Class 8 should be cloudy"

    def test_cloud_shadow(self, synthetic_timestack_with_scl):
        """SCL class 3 (cloud shadow) should be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 3.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 1).all(), "Class 3 (shadow) should be cloudy"

    def test_thin_cirrus(self, synthetic_timestack_with_scl):
        """SCL class 10 (thin cirrus) should be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 10.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 1).all(), "Class 10 (cirrus) should be cloudy"

    def test_vegetation_is_clear(self, synthetic_timestack_with_scl):
        """SCL class 4 (vegetation) should NOT be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 4.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 0).all(), "Class 4 (vegetation) should be clear"

    def test_water_is_clear(self, synthetic_timestack_with_scl):
        """SCL class 6 (water) should NOT be flagged."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 6.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 0).all(), "Class 6 (water) should be clear"

    def test_snow_ice_is_clear(self, synthetic_timestack_with_scl):
        """SCL class 11 (snow/ice) should NOT be flagged as cloud."""
        ts = synthetic_timestack_with_scl.copy()
        ts.loc[dict(band="SCL")] = 11.0

        mask = cloud_pix_mask(ts, method="scl")

        assert (mask == 0).all(), "Class 11 (snow/ice) should be clear"

    def test_missing_scl_raises(self, synthetic_timestack):
        """Should raise ValueError if SCL band is not present."""
        with pytest.raises(ValueError, match="SCL"):
            cloud_pix_mask(synthetic_timestack, method="scl")


class TestCloudPixMaskGeneral:
    """Tests for cloud_pix_mask error handling."""

    def test_invalid_method_raises(self, synthetic_timestack):
        """Should raise ValueError for unknown method names."""
        with pytest.raises(ValueError, match="Invalid"):
            cloud_pix_mask(synthetic_timestack, method="not_a_real_method")

    def test_output_is_named_cloudmask(self, synthetic_timestack_with_swir):
        """Output DataArray should have name='cloudmask'."""
        mask = cloud_pix_mask(synthetic_timestack_with_swir, method="williamson")
        assert mask.name == "cloudmask"
