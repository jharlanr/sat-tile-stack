"""
Utility functions for satellite tile processing.

Provides normalization and cloud masking utilities.
"""

import numpy as np


def cloud_pix_mask(timestack, mask_method):
    """
    Compute a pixel-wise cloud mask over time.

    Parameters
    ----------
    timestack : xarray.DataArray
        Input data with shape [time, band, y, x]. Must include "B11" band.
    mask_method : str
        Masking method to use. Currently supports only "Williamson2018b".

    Returns
    -------
    cloud_mask : xarray.DataArray
        Binary mask with shape [time, y, x]; 0 = clear, 1 = cloudy.

    Raises
    ------
    ValueError
        If an invalid mask_method is provided.
    """
    if mask_method == "Williamson2018b":
        timestack_swir1 = timestack.sel(band="B11")  # shape: [time, y, x]
        is_cloudy = (timestack_swir1 / 10000 > 0.140) | np.isnan(timestack_swir1)
        mask_cloudypix = is_cloudy.astype("uint8")
        mask_cloudypix.name = "cloudmask"
        return mask_cloudypix
    else:
        raise ValueError(
            f"Invalid mask_method '{mask_method}'. "
            f"Supported methods: ['Williamson2018b']"
        )


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
