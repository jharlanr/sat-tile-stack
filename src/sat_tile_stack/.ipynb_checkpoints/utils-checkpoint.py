# utils.py

import numpy as np

## FUNCTION FOR NORMALIZING IMAGERY BEFORE STACKING
def combo_scaler(x, range_max=1):
    median_x = np.nanmedian(x)
    iqr_x = np.nanpercentile(x,75) - np.nanpercentile(x,25)
    robust_x = ((x-median_x)/iqr_x)
    return ((robust_x - np.nanmin(robust_x)) / (np.nanmax(robust_x) - np.nanmin(robust_x))) * range_max