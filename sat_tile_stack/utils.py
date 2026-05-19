"""
Utility functions for satellite tile processing.

Provides normalization and cloud masking utilities.
"""

import numpy as np


CLOUD_METHODS = ["williamson", "scl"]

# Sentinel-2 Scene Classification Layer (SCL) class codes treated as "cloudy"
# for the ESSD `cloud_mask` data variable. This is the single source of truth
# for the SCL cloud definition; `stack.py` and the `cloud_mask` variable's
# `source` attribute both reference it so the dataset's documented semantics
# can never drift from the code.
#
#   3  = cloud shadow
#   8  = cloud, medium probability
#   9  = cloud, high probability
#   10 = thin cirrus
#
# (SCL also defines 0=no_data, 1=saturated/defective, 2=dark area, 4=vegetation,
#  5=bare soil, 6=water, 7=unclassified, 11=snow/ice — none flagged as cloud.)
SCL_CLOUDY_CLASSES = (3, 8, 9, 10)


def cloud_pix_mask(timestack, method="williamson"):
    """
    Compute a pixel-wise cloud mask over time.

    Parameters
    ----------
    timestack : xarray.DataArray
        Input data with shape [time, band, y, x].
    method : str
        Masking method to use:
        - 'williamson': SWIR1 threshold (Williamson2018b). Requires 'B11' band.
          Flags pixels where B11/10000 > 0.140 or B11 is NaN.
        - 'scl': Sentinel-2 Scene Classification Layer. Requires 'SCL' band.
          Flags cloud shadow (3), cloud medium prob (8), cloud high prob (9),
          and thin cirrus (10).

    Returns
    -------
    cloud_mask : xarray.DataArray
        Binary mask with shape [time, y, x]; 0 = clear, 1 = cloudy.

    Raises
    ------
    ValueError
        If an invalid method is provided or required bands are missing.
    """
    if method == "williamson":
        if "B11" not in timestack.band.values:
            raise ValueError("Williamson cloud mask requires 'B11' (SWIR1) band.")
        swir1 = timestack.sel(band="B11")
        is_cloudy = (swir1 / 10000 > 0.140) | np.isnan(swir1)
        mask_out = is_cloudy.astype("uint8")

    elif method == "scl":
        if "SCL" not in timestack.band.values:
            raise ValueError("SCL cloud mask requires 'SCL' band.")
        scl = timestack.sel(band="SCL")
        # 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus
        is_cloudy = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
        mask_out = is_cloudy.astype("uint8")

    else:
        raise ValueError(
            f"Invalid cloud mask method '{method}'. "
            f"Supported: {CLOUD_METHODS}"
        )

    mask_out.name = "cloudmask"
    return mask_out


def combo_scaler(x, p2=75, p1=25, range_max=1):
    """
    Robustly rescale an array using IQR-based normalization.

    Uses median and interquartile range (IQR) for robust normalization,
    then scales to [0, range_max]. NaN values are preserved.

    Parameters
    ----------
    x : array-like
        Input array to normalize.
    p2 : int, optional
        Upper percentile for IQR calculation (default: 75).
    p1 : int, optional
        Lower percentile for IQR calculation (default: 25).
    range_max : float, optional
        Maximum value of the output range (default: 1).

    Returns
    -------
    scaled : numpy.ndarray
        Normalized array with values in [0, range_max].
    """
    x = np.asarray(x)
    with np.errstate(all="ignore"):
        median_x = np.nanmedian(x)
        iqr = np.nanpercentile(x, p2) - np.nanpercentile(x, p1)
        robust = (x - median_x) / iqr
        mn, mx = np.nanmin(robust), np.nanmax(robust)
        scaled = (robust - mn) / (mx - mn) * range_max
    return scaled
