"""
Satellite tile stacking module.

Provides functionality to build multi-band image time-stacks from
satellite imagery via STAC catalogs (e.g., Microsoft Planetary Computer,
Element84 Earth Search).

Supports any STAC collection: Sentinel-2, Sentinel-1, Landsat, etc.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.diagnostics
import pyproj
import stackstac

# Native resolutions (meters) for known STAC collections.
# Used to warn when pix_res is finer than the sensor supports.
NATIVE_RESOLUTIONS = {
    "sentinel-2-l2a": 10,
    "sentinel-1-grd": 20,  # pixel spacing is 10m but true resolution ~20m
    "landsat-c2-l2": 30,
}

from .bounds import (
    sat_mask_array,
    bounds_latlon_around,
    best_crs_for_point,
    pctnanpix_inmask,
    pctcloudypix_inmask,
)
from .utils import combo_scaler, cloud_pix_mask, SCL_CLOUDY_CLASSES

# The only non-spectral asset the ESSD build fetches. Pulled to derive
# `cloud_mask`, then dropped from `reflectance` (it is a classification
# layer, not surface reflectance).
NONSPECTRAL_ASSETS = {"SCL"}

# Sentinel-2 L2A quantification value: surface_reflectance = (DN + offset)/QV.
# Stored in the dataset so the conversion is recoverable from the file alone.
S2_QUANTIFICATION_VALUE = 10000


def _boa_add_offset(item, band_names):
    """BOA additive offset for a STAC item's reflectance assets.

    Sentinel-2 L2A processing baseline >= 04.00 stores DN shifted by -1000;
    surface_reflectance = (DN + boa_add_offset) / quantification_value. The
    offset is product metadata (set by ESA's processing-software version),
    so it is read from the item rather than assumed. Falls back to the
    processing-baseline rule when explicit raster:bands metadata is absent.
    """
    for b in band_names:
        if b in NONSPECTRAL_ASSETS:
            continue
        asset = item.assets.get(b)
        if asset is None:
            continue
        rb = asset.extra_fields.get("raster:bands")
        if rb and isinstance(rb, list) and rb and "offset" in rb[0]:
            try:
                return float(rb[0]["offset"])
            except (TypeError, ValueError):
                pass
        break
    bl = str(item.properties.get("s2:processing_baseline", "")).strip()
    try:
        return -1000.0 if float(bl) >= 4.0 else 0.0
    except ValueError:
        return 0.0


# Per-band STAC metadata we keep as tidy band-indexed coords (units + dtype
# normalized so they survive as coords rather than being flattened to attrs).
_BAND_META_NUMERIC = {
    "gsd": ("m", "nominal ground sample distance"),
    "center_wavelength": ("um", "band center wavelength"),
    "full_width_half_max": ("um", "band full width at half maximum"),
}
_BAND_META_STR = {
    "common_name": "STAC common band name",
    "title": "STAC band title",
}


def _tidy_coords(ds, epsg):
    """Trim stackstac's coordinate clutter to a clean CF coordinate set.

    stackstac attaches every STAC item property as a coordinate. Keep:
    dimension coords (band/x/y/time), the per-band metadata, and the
    time-indexed provenance/QA. Demote every *scalar* item-property coord
    (s2:*, proj:*, epsg, instruments, ...) to a global attribute and drop it
    as a coordinate — they describe the acquisition, not an axis, and CF does
    not want them as coordinate variables. Per-band metadata dtypes/units are
    normalized so they persist as coords (object dtype would otherwise be
    yanked into attrs on write).

    CF-1.8: a coordinate variable must be numeric & monotonic, so the
    string-labelled ``band`` axis is converted to an integer index and the
    short names ('B04'...) move to an auxiliary ``band_name`` coordinate.
    """
    if "band" in ds.coords and ds["band"].dtype.kind in ("U", "S", "O"):
        names = np.array([str(b) for b in ds["band"].values])
        ds = ds.assign_coords(band=np.arange(names.size, dtype="int32"))
        ds = ds.assign_coords(band_name=("band", names))
        ds["band"].attrs.update(long_name="spectral band index", units="1")
        ds["band_name"].attrs.setdefault(
            "long_name", "Sentinel-2 band short name")

    for c in list(ds.coords):
        if c in ("band", "x", "y", "time"):
            continue
        dims = ds[c].dims
        if dims == ("band",):
            if c in _BAND_META_NUMERIC:
                units, ln = _BAND_META_NUMERIC[c]
                ds[c] = ds[c].astype("float64")
                ds[c].attrs.update(units=units, long_name=ln)
            elif c in _BAND_META_STR:
                ds[c] = ds[c].astype(str)
                ds[c].attrs.setdefault("long_name", _BAND_META_STR[c])
            continue
        if dims == ("time",):
            continue  # eo_cloud_cover / pct_nans / processing_baseline / boa_add_offset
        if dims == ():
            val = ds[c].values
            try:
                val = val.item()
            except (ValueError, AttributeError):
                pass
            if isinstance(val, set):
                val = sorted(val)
            ds.attrs.setdefault(c, val)
            ds = ds.reset_coords(c, drop=True)
    # Authoritative CRS hint for finalize_cf's grid_mapping (set after the
    # `epsg` scalar coord is demoted, so it can't be lost).
    ds.attrs.setdefault("crs", f"EPSG:{epsg}")
    return ds


def sattile_stack(
    catalog,
    centroid,
    band_names,
    collection="sentinel-2-l2a",
    pix_res=10,
    tile_size=1024,
    time_range="2019-05-01/2019-09-30",
    cadence="D",
    aggregation="mean",
    normalize=True,
    cloudmask=False,
    query=None,
    mask=None,
    pull_to_mem=False,
):
    """
    Generate a multi-band image time-stack centered on a point.

    This function:
      1. Searches a STAC catalog for all scenes in `collection` covering
         a square around `centroid` between the dates in `time_range`.
      2. Builds a 4D xarray.DataArray (time, band, y, x) using stackstac.
      3. Resamples to the requested temporal `cadence`, carrying forward
         cloud-cover metadata when available.
      4. Optionally applies a robust per-band normalization (to [0, 1]).
      5. Crops to a square "tile" of `tile_size x tile_size` pixels around
         the centroid.
      6. Optionally computes into memory (with a dask progress bar).

    Parameters
    ----------
    catalog : pystac_client.Client
        STAC catalog client (e.g., Planetary Computer, Element84).
    centroid : tuple of float
        (longitude, latitude) in decimal degrees of the point of interest.
    band_names : list of str
        Band/asset names for the target collection
        (e.g. ['B04','B03','B02'] for S2, ['VV','VH'] for S1,
        ['SR_B4','SR_B3','SR_B2'] for Landsat).
    collection : str, optional
        STAC collection ID to search (default: 'sentinel-2-l2a').
    pix_res : int, optional
        Pixel resolution in meters (default: 10).
    tile_size : int, optional
        Size of the output tile in pixels (square) (default: 1024).
    time_range : str, optional
        ISO8601 date range "YYYY-MM-DD/YYYY-MM-DD" for imagery search
        (default: '2019-05-01/2019-09-30').
    cadence : str, optional
        Temporal resampling frequency as a pandas offset alias.
        Examples: 'D' (daily), '2D' (every 2 days), 'W' (weekly),
        'MS' (month start), 'ME' (month end). Default: 'D'.
    aggregation : str, optional
        How to combine multiple observations within each cadence window.
        Options:
        - 'mean': average all observations (default). Can blend scenes.
        - 'nearest': pick the observation closest to each cadence timestep.
          Preserves individual scenes with no blending.
        - 'first': take the first observation in each window.
        - 'last': take the last observation in each window.
    normalize : bool, optional
        If True, apply a robust (median/IQR) normalization to each band
        so values are scaled into [0,1] (default: True).
    cloudmask : str, callable, or False, optional
        Cloud masking strategy. Options:
        - False: no cloud mask (default)
        - 'scl': Sentinel-2 Scene Classification Layer (requires 'SCL' band).
          Flags cloud shadow, medium/high probability cloud, and thin cirrus.
        - 'williamson': SWIR1 threshold method (requires 'B11' band).
        - callable: custom function that takes a timestack DataArray and
          returns a DataArray of shape (time, y, x) with 0=clear, 1=cloudy.
    query : dict, optional
        STAC query filter. Default: {"eo:cloud_cover": {"lt": 100}} for
        optical collections. Set to {} to disable filtering.
    mask : geopandas.GeoDataFrame or None, optional
        If not None, generate a spatial mask from the GeoDataFrame and
        append to DataArray (default: None).
    pull_to_mem : bool, optional
        If True, triggers `timestack.compute()` and returns an in-memory
        xarray.DataArray; otherwise returns a lazy dask-backed DataArray
        (default: False).

    Returns
    -------
    xarray.Dataset
        Dataset with data variables:
          - reflectance (time, band, y, x): spectral bands only (SCL excluded),
            raw L2A DN, float32. NOT normalized.
          - cloud_mask (time, y, x): uint8 SCL-derived cloud flag
            (0=clear, 1=cloudy), present when 'SCL' is in `band_names`.
        Coordinates include time, band (spectral), y, x, eo_cloud_cover,
        pct_nans, and the per-timestep processing provenance
        `processing_baseline` and `boa_add_offset` (the raw->surface-
        reflectance recipe; see `reflectance` attrs). The legacy `cloudmask`
        method arg is retained for signature compatibility but the cloud
        mask is always SCL-derived; `mask` is ignored (see warning).

    Notes
    -----
    - Requires `stackstac`, `xarray`, `dask`, and a STAC `catalog` in scope.
    - CRS for reprojection is taken from the first item's projection extension,
      falling back to best_crs_for_point() if unavailable.
    """
    # Search imagery catalog for items matching location and date range
    bounds_latlon = bounds_latlon_around(*centroid, side_m=pix_res * tile_size * 1.1)

    search_kwargs = {
        "collections": [collection],
        "bbox": bounds_latlon,
        "datetime": time_range,
    }
    if query is not None:
        search_kwargs["query"] = query
    else:
        search_kwargs["query"] = {"eo:cloud_cover": {"lt": 100}}

    try:
        search = catalog.search(**search_kwargs)
        items = search.item_collection()
    except Exception:
        # If cloud_cover query fails (e.g., SAR collection), retry without it
        if query is None:
            search_kwargs.pop("query", None)
            search = catalog.search(**search_kwargs)
            items = search.item_collection()
        else:
            raise

    if len(items) == 0:
        raise ValueError(
            f"No items found for collection='{collection}', "
            f"centroid={centroid}, time_range='{time_range}'"
        )

    # Warn if requested resolution is finer than native
    if collection in NATIVE_RESOLUTIONS and pix_res < NATIVE_RESOLUTIONS[collection]:
        native = NATIVE_RESOLUTIONS[collection]
        print(
            f"WARNING: Requested pix_res={pix_res}m is finer than the native "
            f"resolution of {collection} ({native}m). The output will be "
            f"interpolated — no additional detail beyond {native}m."
        )

    # Determine CRS: prefer projection extension, fall back to best_crs_for_point
    try:
        from pystac.extensions.projection import ProjectionExtension as proj_ext
        epsg = proj_ext.ext(items[0]).epsg
    except Exception:
        crs = best_crs_for_point(*centroid)
        epsg = crs.to_epsg()

    # Create the stack
    stack = stackstac.stack(
        items,
        epsg=epsg,
        assets=band_names,
        bounds_latlon=bounds_latlon,
        resolution=pix_res,
        chunksize=(1, 1, 4096, 4096),
        fill_value=np.nan,
    )

    # Mask out black 0.0 pixels as NaNs
    nodata_mask = (stack == 0).all(dim="band")
    stack = stack.where(~nodata_mask)

    # Split spectral reflectance from the SCL classification BEFORE resampling.
    # Spectral bands are continuous and may be aggregated; SCL is categorical
    # (class codes) so averaging it is meaningless — it is resampled with a
    # categorical-safe 'first' regardless of `aggregation`. This is a
    # deliberate correctness fix over the legacy single-array path, where a
    # mean-resampled SCL could produce fractional codes that match no class.
    spectral_names = [b for b in band_names if b not in NONSPECTRAL_ASSETS]
    has_scl = "SCL" in band_names
    spec = stack.sel(band=spectral_names)
    scl = stack.sel(band="SCL") if has_scl else None

    # Resample spectral reflectance to requested cadence
    start, end = time_range.split("/")
    full_steps = pd.date_range(start, end, freq=cadence)

    if aggregation == "mean":
        spec_res = spec.resample(time=cadence).mean("time", keep_attrs=True)
    elif aggregation == "nearest":
        spec_res = spec.reindex(time=full_steps, method="nearest", tolerance=cadence)
    elif aggregation == "first":
        spec_res = spec.resample(time=cadence).first(keep_attrs=True)
    elif aggregation == "last":
        spec_res = spec.resample(time=cadence).last(keep_attrs=True)
    else:
        raise ValueError(
            f"Invalid aggregation '{aggregation}'. "
            f"Supported: 'mean', 'nearest', 'first', 'last'"
        )

    spec_res = spec_res.reindex(time=full_steps, fill_value=np.nan)
    stack_resampled = spec_res

    # Resample SCL categorically (first observation in each cadence bin).
    if scl is not None:
        scl_res = scl.resample(time=cadence).first(keep_attrs=True)
        scl_res = scl_res.reindex(time=full_steps, fill_value=np.nan)
    else:
        scl_res = None

    # Extract cloud cover metadata if available
    if items and "eo:cloud_cover" in items[0].properties:
        cc_da = xr.DataArray(
            np.array([item.properties["eo:cloud_cover"] for item in items]),
            coords={"time": stack.time},
            dims=["time"],
        )
        daily_cc = cc_da.resample(time=cadence).max()
        daily_cc = daily_cc.reindex(time=full_steps, fill_value=np.nan)
        stack_resampled = stack_resampled.assign_coords(eo_cloud_cover=daily_cc)

    # --- Per-timestep Sentinel-2 processing provenance ---
    # `processing_baseline` (ESA software version) and `boa_add_offset` travel
    # with the *product*, not the acquisition date. Recorded per original scene
    # and aligned onto the resampled timeline so the raw->reflectance recipe
    # (surface_reflectance = (DN + boa_add_offset) / quantification_value) is
    # recoverable from the file alone. Categorical-safe 'first' per cadence bin.
    baselines = np.array(
        [str(it.properties.get("s2:processing_baseline", "")) for it in items],
        dtype=object,
    )
    offsets = np.array(
        [_boa_add_offset(it, band_names) for it in items], dtype="float64"
    )
    bl_da = xr.DataArray(baselines, coords={"time": stack.time}, dims=["time"])
    off_da = xr.DataArray(offsets, coords={"time": stack.time}, dims=["time"])
    bl_daily = bl_da.resample(time=cadence).first().reindex(
        time=full_steps, fill_value=""
    )
    # Empty resample bins yield NaN (float) even for an object/string array;
    # normalize the baseline coord to clean strings ("" == no observation) so
    # it is both NetCDF-writable and safely sortable downstream.
    # Fixed-width unicode (NOT object): an object-dtype coord is demoted to a
    # global attr by io.sanitise_dataset, which would silently drop
    # processing_baseline as a per-timestep coordinate.
    bl_clean = np.array(
        [b if isinstance(b, str) else "" for b in bl_daily.values],
        dtype="<U16",
    )
    bl_daily = bl_daily.copy(data=bl_clean)
    off_daily = off_da.resample(time=cadence).first().reindex(
        time=full_steps, fill_value=np.nan
    )
    stack_resampled = stack_resampled.assign_coords(
        processing_baseline=bl_daily, boa_add_offset=off_daily
    )

    # Apply normalization if requested
    if normalize:
        stack_rechunk = stack_resampled.chunk({"y": -1, "x": -1})
        stack_resampled = xr.apply_ufunc(
            combo_scaler,
            stack_rechunk,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"range_max": 1},
            keep_attrs=True,
        )

    if mask is not None:
        warnings.warn(
            "`mask` is ignored by sattile_stack in the v2 schema. The per-lake "
            "footprint (`lake_boundary`) is co-registered separately via "
            "coregister.add_static_polygon (Dunmire 2021).",
            stacklevel=2,
        )

    # Crop tiles to desired size (reflectance + SCL share the same grid)
    buffer = tile_size * pix_res / 2  # [m]
    x_utm, y_utm = pyproj.Proj(stack.crs)(*centroid)
    reflectance_ts = stack_resampled.loc[
        ..., y_utm + buffer : y_utm - buffer, x_utm - buffer : x_utm + buffer
    ].astype("float32")

    scl_ts = None
    if scl_res is not None:
        scl_ts = scl_res.loc[
            ..., y_utm + buffer : y_utm - buffer, x_utm - buffer : x_utm + buffer
        ]

    # Track percent of NaN pixels in the spectral tile
    nan_counts = reflectance_ts.isnull().sum(dim=("band", "y", "x"))
    total = len(spectral_names) * tile_size * tile_size
    pct_nans = (nan_counts / total) * 100
    reflectance_ts = reflectance_ts.assign_coords(pct_nans=("time", pct_nans.values))

    # Assemble the CF-style Dataset: reflectance (spectral) + cloud_mask (SCL).
    ds = xr.Dataset(
        data_vars={"reflectance": reflectance_ts},
        attrs=dict(stack_resampled.attrs),
    )
    if scl_ts is not None:
        cloud_mask = scl_ts.isin(SCL_CLOUDY_CLASSES).astype("uint8")
        cloud_mask = cloud_mask.drop_vars("band", errors="ignore")
        cloud_mask.attrs = {
            "long_name": "Sentinel-2 SCL-derived cloud flag",
            "flag_values": np.array([0, 1], dtype="uint8"),
            "flag_meanings": "clear cloudy",
            "source": (
                "Sentinel-2 L2A Scene Classification Layer (SCL); cloudy = SCL "
                f"classes {tuple(int(c) for c in SCL_CLOUDY_CLASSES)} "
                "(3=cloud_shadow, 8=cloud_medium_prob, 9=cloud_high_prob, "
                "10=thin_cirrus). Categorical 'first' resampling per cadence "
                "bin. Days with no observation are 0 (clear); consult "
                "`pct_nans` / reflectance NaNs to distinguish no-data."
            ),
        }
        ds["cloud_mask"] = cloud_mask

    # Demote stackstac's scalar item-property coords to global attrs; keep a
    # clean CF coordinate set (band/x/y/time + band metadata + time provenance).
    ds = _tidy_coords(ds, epsg)

    # Raw->surface-reflectance recipe, recoverable from the file alone.
    ds["reflectance"].attrs.setdefault(
        "long_name", "Sentinel-2 L2A raw surface-reflectance digital numbers"
    )
    ds["reflectance"].attrs.setdefault("units", "1")
    ds["reflectance"].attrs.setdefault(
        "comment",
        "Raw Sentinel-2 L2A (BOA) digital numbers as delivered by Microsoft "
        "Planetary Computer; NOT normalized. To convert to surface "
        "reflectance: surface_reflectance = (DN + boa_add_offset) / "
        f"quantification_value, with quantification_value = "
        f"{S2_QUANTIFICATION_VALUE}. `boa_add_offset` and `processing_baseline` "
        "are provided per timestep. Normalization is a downstream (ML) choice.",
    )
    ds.attrs["s2:quantification_value"] = S2_QUANTIFICATION_VALUE

    # Mixed-baseline checkpoint: surface it in attrs (stdout is suppressed in
    # batch builds) and warn for interactive callers.
    fin = off_daily.values[np.isfinite(off_daily.values.astype("float64"))] \
        if off_daily.size else np.array([])
    uniq_off = sorted({float(v) for v in fin})
    uniq_bl = sorted({str(b) for b in bl_daily.values.tolist()
                      if isinstance(b, str) and b})
    ds.attrs["s2:processing_baseline_values"] = uniq_bl
    ds.attrs["boa_add_offset_values"] = uniq_off
    mixed = len(uniq_off) > 1
    ds.attrs["mixed_processing_baseline"] = int(mixed)  # netCDF has no bool
    if mixed:
        warnings.warn(
            f"Stack mixes Sentinel-2 processing baselines {uniq_bl} "
            f"(boa_add_offset {uniq_off}). Raw DN are stored unaltered; "
            "per-timestep `boa_add_offset` must be applied before comparing "
            "values across the time series.",
            stacklevel=2,
        )

    if pull_to_mem:
        print(
            f"Pulling stack into memory: {len(items)} scenes -> "
            f"vars {list(ds.data_vars)}, dims {dict(ds.sizes)} "
            f"(this is network/IO-bound; minutes per full-season 512^2 lake)",
            flush=True,
        )
        with dask.diagnostics.ProgressBar():
            ds = ds.compute()
        # Distribution sanity gate (catches gross scale/offset corruption
        # before a full Sherlock run). Raw S2 DN of glacial scenes sit well
        # within (0, 20000); a wildly out-of-range median means something
        # upstream is wrong.
        rv = ds["reflectance"].values
        finite = np.isfinite(rv)
        if finite.any():
            med = float(np.nanmedian(rv[finite]))
            if not (0.0 < med < 20000.0):
                raise ValueError(
                    f"reflectance median DN {med:.1f} outside plausible "
                    f"(0, 20000) — suspect scale/offset corruption "
                    f"(baselines={uniq_bl}, offsets={uniq_off})."
                )
        print(f"Stack loaded, vars: {list(ds.data_vars)}", flush=True)

    return ds
