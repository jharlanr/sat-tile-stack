# utils.py

import numpy as np
import warnings

## FUNCTION FOR NORMALIZING IMAGERY BEFORE STACKING
def combo_scaler(x, range_min=0.01, range_max=1.0, dtype=np.float32):
    """
    Robustly rescale an image / stack so that valid data lie in [range_min, range_max].
    • NaNs are preserved as NaNs.
    • RuntimeWarnings from all-NaN slices are locally silenced.
    """
    arr = np.asarray(x, dtype=dtype)

    # ─── Silences only the specific RuntimeWarning we care about ───
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=RuntimeWarning,
                                message="All-NaN slice encountered")

        with np.errstate(all="ignore"):          # suppress FP warnings
            if np.isnan(arr).all():              # whole slice is NaN → return as-is
                return arr.copy()

            med = np.nanmedian(arr)
            q75 = np.nanpercentile(arr, 75)
            q25 = np.nanpercentile(arr, 25)
            iqr = q75 - q25                      # robust spread

    # No spread or numerical trouble → pass data through unchanged
    if iqr == 0 or np.isnan(iqr):
        return arr.copy()

    robust = (arr - med) / iqr                  # robust standardisation

    with np.errstate(all="ignore"):
        mn, mx = np.nanmin(robust), np.nanmax(robust)

    if mx == mn or np.isnan(mn) or np.isnan(mx):
        return arr.copy()

    scale  = (range_max - range_min) / (mx - mn)
    scaled = (robust - mn) * scale + range_min

    return scaled.astype(dtype)

# def combo_scaler(x, range_max=1):
#     x = np.asarray(x)
#     with np.errstate(all="ignore"):
#         median_x = np.nanmedian(x)
#         iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
#         robust = (x - median_x) / iqr
#         mn, mx = np.nanmin(robust), np.nanmax(robust)
#         scaled = (robust - mn) / (mx - mn) * range_max
#     return scaled


