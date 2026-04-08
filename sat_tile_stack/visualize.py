"""
Visualization utilities for satellite tile timestacks.

Provides functions for creating movies, static frames, and multi-panel
visualizations from timestack DataArrays or NetCDF files.

Supports any sensor (Sentinel-2, Sentinel-1, Landsat) with configurable
band selection, scaling, and overlays.

Examples
--------
Quick movie from a NetCDF file:

    >>> timestack_to_movie("tstack_lake42.nc")

Custom false-color movie:

    >>> timestack_to_movie(stack, bands=["B11", "B08", "B04"], scale="percentile")

Movie with dynamic (per-timestep) mask overlay:

    >>> timestack_to_movie(stack, mask_band="water_mask", mask_mode="dynamic")

Static frame export:

    >>> export_frame(stack, time_index="2019-06-10", outfile="frame.png")

Multi-panel comparison:

    >>> multi_panel_frame(stack, time_index=5, panels=[
    ...     {"bands": ["B04","B03","B02"], "label": "True Color"},
    ...     {"bands": ["B11","B08","B04"], "label": "SWIR False Color"},
    ...     {"bands": ["cloudmask"], "label": "Cloud Mask", "cmap": "RdYlGn_r"},
    ... ])

Batch processing:

    >>> batch_movies("data/processed/", fps=4, scale="percentile")
"""

import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from pathlib import Path


# ===========================================================================
# Scaling functions — convert raw pixel values to [0, 1] for display
# ===========================================================================

def _scale_divide(rgb, factor=10000.0):
    """Simple division scaling (e.g., S2 reflectance / 10000)."""
    return np.clip(rgb / factor, 0, 1)


def _scale_percentile(rgb, p_low=2, p_high=98):
    """Per-channel percentile stretch to [0, 1]. Robust to outliers."""
    out = np.zeros_like(rgb, dtype=np.float64)
    for ch in range(rgb.shape[-1]):
        band = rgb[:, :, ch]
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, [p_low, p_high])
        out[:, :, ch] = (band - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1)


def _scale_minmax(rgb):
    """Per-channel min-max scaling to [0, 1]."""
    out = np.zeros_like(rgb, dtype=np.float64)
    for ch in range(rgb.shape[-1]):
        band = rgb[:, :, ch]
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            continue
        lo, hi = np.nanmin(valid), np.nanmax(valid)
        out[:, :, ch] = (band - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1)


SCALE_METHODS = {
    "divide": _scale_divide,
    "percentile": _scale_percentile,
    "minmax": _scale_minmax,
    "none": lambda rgb, **kw: np.clip(rgb, 0, 1),
}


# ===========================================================================
# Helper: load data from file or DataArray
# ===========================================================================

def _load_data(source, var="reflectance"):
    """Load a DataArray from a file path or pass through if already a DataArray."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        ds = xr.open_dataset(path)
        return ds[var], path
    elif isinstance(source, xr.DataArray):
        return source, None
    else:
        raise TypeError(f"Expected file path or DataArray, got {type(source)}")


def _get_source_name(source, path):
    """Extract a human-readable name from the source."""
    if path is not None:
        return path.stem.replace("tstack_", "")
    name = source.attrs.get("lake_id") or source.attrs.get("lake_name") or ""
    return name


# ===========================================================================
# Core rendering: build an RGB frame from a single timestep
# ===========================================================================

def _render_frame(tslice, bands, scale, scale_kwargs):
    """
    Render a single timestep as an RGB numpy array [H, W, 3].

    Parameters
    ----------
    tslice : xarray.DataArray
        Single timestep with dims (band, y, x).
    bands : list of str
        Three band names for R, G, B channels.
    scale : str or callable
        Scaling method name or custom function.
    scale_kwargs : dict
        Extra kwargs passed to the scaling function.

    Returns
    -------
    numpy.ndarray
        RGB image [H, W, 3] with values in [0, 1].
    """
    channels = []
    for b in bands:
        if b in tslice.band.values:
            channels.append(tslice.sel(band=b).values)
        else:
            raise ValueError(f"Band '{b}' not found. Available: {list(tslice.band.values)}")

    rgb = np.stack(channels, axis=-1)  # [H, W, 3]

    # Apply scaling
    if callable(scale):
        rgb = scale(rgb, **scale_kwargs)
    elif scale in SCALE_METHODS:
        rgb = SCALE_METHODS[scale](rgb, **scale_kwargs)
    else:
        raise ValueError(f"Unknown scale '{scale}'. Options: {list(SCALE_METHODS.keys())} or a callable")

    return np.nan_to_num(rgb, nan=0).astype(np.float64)


# ===========================================================================
# Mask overlay helper
# ===========================================================================

def _draw_mask_overlay(ax, da, tslice, mask_band, mask_mode, mask_color, time_idx=None):
    """
    Draw a mask contour overlay on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    da : xarray.DataArray
        Full timestack (used for static mask — takes first timestep).
    tslice : xarray.DataArray
        Current timestep slice (used for dynamic mask).
    mask_band : str or None
        Band name for the mask. None to skip.
    mask_mode : str
        'static' — same mask every frame (uses the first non-NaN timestep).
        'dynamic' — mask varies per timestep.
    mask_color : str
        Contour color.
    time_idx : int, optional
        Current time index (for static mode, used on first call only).
    """
    if mask_band is None:
        return
    if mask_band not in tslice.band.values:
        return

    if mask_mode == "dynamic":
        mask2d = tslice.sel(band=mask_band).values
    elif mask_mode == "static":
        # Use the first timestep that has valid mask data
        for t in range(len(da.time)):
            candidate = da.isel(time=t).sel(band=mask_band).values
            if not np.isnan(candidate).all():
                mask2d = candidate
                break
        else:
            return  # no valid mask found
    else:
        raise ValueError(f"mask_mode must be 'static' or 'dynamic', got '{mask_mode}'")

    if not np.isnan(mask2d).all():
        ax.contour(mask2d, levels=[0.5], colors=mask_color, linewidths=1)


# ===========================================================================
# export_frame — save a single timestep as PNG
# ===========================================================================

def export_frame(
    source,
    time_index=0,
    bands=("B04", "B03", "B02"),
    scale="divide",
    scale_kwargs=None,
    mask_band="mask",
    mask_mode="static",
    mask_color="red",
    title=None,
    figsize=(6, 6),
    outfile=None,
    dpi=150,
    var="reflectance",
):
    """
    Export a single timestep as a static image (PNG).

    Parameters
    ----------
    source : DataArray, str, or Path
        Timestack or path to a NetCDF file.
    time_index : int or str
        Which timestep to display (integer index or date string).
    bands : tuple of str
        Three band names for RGB display.
    scale : str or callable
        Scaling method: 'divide' (default, /10000), 'percentile' (2-98%),
        'minmax', 'none', or a custom callable(rgb, **kwargs) -> rgb.
    scale_kwargs : dict, optional
        Extra kwargs for the scaling function (e.g., {'factor': 5000}).
    mask_band : str or None
        Band name to contour as overlay. None to skip.
    mask_mode : str
        'static' (same mask every frame) or 'dynamic' (per-timestep mask).
    mask_color : str
        Color for mask contour (default: 'red').
    title : str, optional
        Custom title. If None, auto-generates from date and metadata.
    figsize : tuple
        Figure size in inches (default: (6, 6)).
    outfile : str or Path, optional
        Output file path. If None, displays interactively.
    dpi : int
        Resolution (default: 150).
    var : str
        Variable name in NetCDF file (default: 'reflectance').
    """
    scale_kwargs = scale_kwargs or {}
    da, path = _load_data(source, var=var)

    # Select timestep
    if isinstance(time_index, str):
        tslice = da.sel(time=time_index).load()
    else:
        tslice = da.isel(time=time_index).load()

    # Render RGB
    rgb = _render_frame(tslice, list(bands), scale, scale_kwargs)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.axis("off")

    # Mask contour overlay
    _draw_mask_overlay(ax, da, tslice, mask_band, mask_mode, mask_color)

    # Title
    if title is None:
        name = _get_source_name(source, path)
        date = np.datetime_as_string(tslice.time.values, unit="D")
        parts = [name, date]
        if "eo_cloud_cover" in tslice.coords:
            cc = tslice["eo_cloud_cover"].values
            if not np.isnan(cc):
                parts.append(f"Cloud: {float(cc):.1f}%")
        if "pct_nans" in tslice.coords:
            pn = tslice["pct_nans"].values
            if not np.isnan(pn):
                parts.append(f"NaN: {float(pn):.1f}%")
        title = "  ".join(parts)
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved frame: {outfile}")
    else:
        plt.show()


# ===========================================================================
# timestack_to_movie — animated movie from timestack
# ===========================================================================

def timestack_to_movie(
    source,
    bands=("B04", "B03", "B02"),
    scale="divide",
    scale_kwargs=None,
    mask_band="mask",
    mask_mode="static",
    mask_color="red",
    fps=6,
    figsize=(6, 6),
    outfile=None,
    dpi=150,
    var="reflectance",
    skip_empty=False,
):
    """
    Build a movie from a timestack.

    Parameters
    ----------
    source : DataArray, str, or Path
        Timestack or path to a NetCDF file.
    bands : tuple of str
        Three band names for RGB display. For false color, use e.g.
        ('B11', 'B08', 'B04') for SWIR/NIR/Red.
    scale : str or callable
        Scaling method: 'divide' (default, /10000), 'percentile' (2-98%),
        'minmax', 'none', or a custom callable.
    scale_kwargs : dict, optional
        Extra kwargs for scaling (e.g., {'factor': 5000} for divide).
    mask_band : str or None
        Band name to contour as overlay. None to skip.
    mask_mode : str
        'static' — same mask contour every frame (e.g., max extent polygon).
        'dynamic' — mask changes per timestep (e.g., per-scene water detection).
    mask_color : str
        Color for mask contour (default: 'red').
    fps : int
        Frames per second (default: 6).
    figsize : tuple
        Figure size in inches (default: (6, 6)).
    outfile : str or Path, optional
        Output file. If None, auto-generates from source path.
    dpi : int
        Resolution (default: 150).
    var : str
        Variable name in NetCDF (default: 'reflectance').
    skip_empty : bool
        If True, skip all-NaN frames (default: False).

    Returns
    -------
    None
        Saves the movie to outfile.
    """
    scale_kwargs = scale_kwargs or {}
    da, path = _load_data(source, var=var)
    name = _get_source_name(source, path)

    # Determine output path
    if outfile is None:
        if path is not None:
            outfile = path.with_suffix(".mp4")
        else:
            outfile = Path(f"{name or 'timestack'}.mp4")
    outfile = Path(outfile)

    times = da.time.values

    # Determine which frames to include
    if skip_empty:
        frame_indices = []
        for i in range(len(times)):
            tslice = da.isel(time=i)
            band_data = tslice.sel(band=list(bands))
            if not np.isnan(band_data.values).all():
                frame_indices.append(i)
    else:
        frame_indices = list(range(len(times)))

    n_frames = len(frame_indices)
    if n_frames == 0:
        print("No frames to render (all empty). Skipping.")
        return

    # Precompute static mask if needed
    static_mask = None
    if mask_band and mask_mode == "static" and mask_band in da.band.values:
        for t in range(len(times)):
            candidate = da.isel(time=t).sel(band=mask_band).values
            if not np.isnan(candidate).all():
                static_mask = candidate
                break

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")

    def render(frame_num):
        ax.clear()
        ax.set_facecolor("black")
        ax.axis("off")

        i = frame_indices[frame_num]
        tslice = da.isel(time=i).load()

        # Check if this frame has data
        band_data = tslice.sel(band=list(bands))
        if np.isnan(band_data.values).all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=14, color="gray", transform=ax.transAxes)
        else:
            # Render RGB
            rgb = _render_frame(tslice, list(bands), scale, scale_kwargs)
            ax.imshow(rgb)

            # Mask contour
            if mask_band and mask_band in tslice.band.values:
                if mask_mode == "dynamic":
                    mask2d = tslice.sel(band=mask_band).values
                elif mask_mode == "static" and static_mask is not None:
                    mask2d = static_mask
                else:
                    mask2d = None

                if mask2d is not None and not np.isnan(mask2d).all():
                    ax.contour(mask2d, levels=[0.5], colors=mask_color, linewidths=1)

        # Title with metadata
        date = np.datetime_as_string(times[i], unit="D")
        parts = [name, date]
        if "eo_cloud_cover" in tslice.coords:
            cc = tslice["eo_cloud_cover"].values
            if not np.isnan(cc):
                parts.append(f"Cloud: {float(cc):.1f}%")
        if "pct_nans" in tslice.coords:
            pn = tslice["pct_nans"].values
            if not np.isnan(pn):
                parts.append(f"NaN: {float(pn):.1f}%")
        ax.set_title("  ".join(parts), color="white", fontsize=9)

    # Create animation
    anim = mpl_anim.FuncAnimation(
        fig, render, frames=n_frames, interval=1000 / fps, blit=False
    )

    # Save
    if outfile.suffix == ".gif":
        anim.save(str(outfile), writer="imagemagick", fps=fps, dpi=dpi)
    else:
        writer = mpl_anim.FFMpegWriter(fps=fps, metadata={"artist": "sat-tile-stack"})
        anim.save(str(outfile), writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved {outfile} ({n_frames} frames, {fps} fps)")


# ===========================================================================
# multi_panel_frame — side-by-side band combinations
# ===========================================================================

def multi_panel_frame(
    source,
    time_index=0,
    panels=None,
    scale="divide",
    scale_kwargs=None,
    mask_band="mask",
    mask_mode="static",
    mask_color="red",
    figsize=None,
    outfile=None,
    dpi=150,
    var="reflectance",
):
    """
    Display multiple band combinations side by side for a single timestep.

    Useful for comparing RGB, false-color, and derived products (e.g., cloud mask).

    Parameters
    ----------
    source : DataArray, str, or Path
        Timestack or path to NetCDF file.
    time_index : int or str
        Which timestep to display.
    panels : list of dict
        Each panel is a dict with:
        - 'bands': list of 3 band names for RGB, or 1 band for grayscale
        - 'label': str, panel title
        - 'scale': str or callable (optional, overrides global scale)
        - 'cmap': str (optional, for single-band grayscale/colormap display)
        If None, defaults to [{'bands': ['B04','B03','B02'], 'label': 'True Color'}].
    scale : str or callable
        Default scaling for all panels.
    scale_kwargs : dict, optional
        Default scale kwargs for all panels.
    mask_band : str or None
        Band to contour on each panel. None to skip.
    mask_mode : str
        'static' or 'dynamic'.
    mask_color : str
        Contour color.
    figsize : tuple, optional
        Figure size. If None, auto-sizes based on number of panels.
    outfile : str or Path, optional
        Save to file. If None, displays interactively.
    dpi : int
        Resolution.
    var : str
        Variable name in NetCDF.

    Example
    -------
    >>> multi_panel_frame(stack, time_index="2019-06-10", panels=[
    ...     {"bands": ["B04", "B03", "B02"], "label": "True Color"},
    ...     {"bands": ["B11", "B08", "B04"], "label": "SWIR False Color"},
    ...     {"bands": ["cloudmask"], "label": "Cloud Mask", "cmap": "RdYlGn_r"},
    ... ])
    """
    scale_kwargs = scale_kwargs or {}
    da, path = _load_data(source, var=var)

    if panels is None:
        panels = [{"bands": ["B04", "B03", "B02"], "label": "True Color"}]

    # Select timestep
    if isinstance(time_index, str):
        tslice = da.sel(time=time_index).load()
    else:
        tslice = da.isel(time=time_index).load()

    n = len(panels)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    name = _get_source_name(source, path)
    date = np.datetime_as_string(tslice.time.values, unit="D")

    for ax, panel in zip(axes, panels):
        panel_bands = panel["bands"]
        panel_scale = panel.get("scale", scale)
        panel_cmap = panel.get("cmap", None)
        label = panel.get("label", ", ".join(panel_bands))

        if panel_cmap and len(panel_bands) == 1:
            # Single-band grayscale/colormap display
            data = tslice.sel(band=panel_bands[0]).values
            ax.imshow(np.nan_to_num(data, nan=0), cmap=panel_cmap)
        else:
            # RGB display
            rgb = _render_frame(tslice, panel_bands, panel_scale, scale_kwargs)
            ax.imshow(rgb)

        # Mask contour
        _draw_mask_overlay(ax, da, tslice, mask_band, mask_mode, mask_color)

        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{name}  {date}", fontsize=12)
    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outfile}")
    else:
        plt.show()


# ===========================================================================
# batch_movies — process a directory of NetCDF files
# ===========================================================================

def batch_movies(
    directory,
    bands=("B04", "B03", "B02"),
    scale="divide",
    scale_kwargs=None,
    mask_band="mask",
    mask_mode="static",
    mask_color="red",
    fps=6,
    dpi=100,
    var="reflectance",
    skip_empty=False,
    pattern="*.nc",
):
    """
    Generate movies for all NetCDF files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing .nc timestack files.
    bands, scale, scale_kwargs, mask_band, mask_mode, mask_color,
    fps, dpi, var, skip_empty
        Passed to timestack_to_movie() for each file.
    pattern : str
        Glob pattern for finding files (default: '*.nc').

    Returns
    -------
    list of Path
        Paths to all generated movie files.
    """
    directory = Path(directory)
    nc_files = sorted(directory.glob(pattern))

    if not nc_files:
        print(f"No files matching '{pattern}' in {directory}")
        return []

    print(f"Processing {len(nc_files)} files from {directory}...")
    outputs = []

    for i, nc_path in enumerate(nc_files):
        print(f"  [{i+1}/{len(nc_files)}] {nc_path.name}")
        try:
            outfile = nc_path.with_suffix(".mp4")
            timestack_to_movie(
                nc_path, bands=bands, scale=scale,
                scale_kwargs=scale_kwargs or {}, mask_band=mask_band,
                mask_mode=mask_mode, mask_color=mask_color,
                fps=fps, dpi=dpi, var=var, skip_empty=skip_empty,
                outfile=outfile,
            )
            outputs.append(outfile)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"Done. Generated {len(outputs)}/{len(nc_files)} movies.")
    return outputs
