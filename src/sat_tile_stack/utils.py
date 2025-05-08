# utils.py

import numpy as np

## FUNCTION FOR NORMALIZING IMAGERY BEFORE STACKING
def combo_scaler(x, range_max=1):
    x = np.asarray(x)
    with np.errstate(all="ignore"):
        median_x = np.nanmedian(x)
        iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
        robust = (x - median_x) / iqr
        mn, mx = np.nanmin(robust), np.nanmax(robust)
        scaled = (robust - mn) / (mx - mn) * range_max
    return scaled