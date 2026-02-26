"""
Visualization utilities for satellite tile timestacks.

Provides functions for creating movies/animations from NetCDF timestacks.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from pathlib import Path


def timestack_to_movie(
    nc_path,
    var="reflectance",
    bands=("B04", "B03", "B02"),
    fps=6,
    outfile=None,
    dpi=150,
):
    """
    Build a quicklook movie from a time x band x y x x NetCDF stack.

    Creates an MP4 or GIF animation from satellite imagery, normalizing
    pixels by 10000, using a black background, and outlining the mask
    (last band) on each frame.

    Parameters
    ----------
    nc_path : str or Path
        Path to the NetCDF file containing the timestack.
    var : str, optional
        Variable name in the NetCDF file (default: "reflectance").
    bands : tuple of str, optional
        Band names for RGB display (default: ("B04", "B03", "B02")).
    fps : int, optional
        Frames per second for the output movie (default: 6).
    outfile : str or Path, optional
        Output file path. If None, uses the input path with .mp4 extension.
    dpi : int, optional
        Resolution of the output movie (default: 150).

    Returns
    -------
    None
        Saves the movie to the specified output file.
    """
    nc_path = Path(nc_path)
    outfile = Path(outfile or nc_path.with_suffix(".mp4"))

    ds = xr.open_dataset(nc_path)
    data = ds[var]  # (time, band, y, x)
    times = data.time.values
    n_frames = len(times)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")

    def frame(i):
        ax.clear()
        ax.set_facecolor("black")
        ax.axis("off")

        # Load one timestep
        tslice = data.isel(time=i).load()

        # Build RGB
        r = tslice.isel(band=0).values / 10000.0
        g = tslice.isel(band=1).values / 10000.0
        b = tslice.isel(band=2).values / 10000.0
        rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
        ax.imshow(rgb)

        # Contour the mask (last band)
        mask2d = tslice.isel(band=-1).values
        ax.contour(mask2d, levels=[0.5], colors="red", linewidths=1)

        # Title with lake name, date, clouds, nans
        date = np.datetime_as_string(times[i], unit="D")
        lake_name = nc_path.stem.replace("tstack_", "")
        pct_nan = float(tslice["pct_nans"].values)
        pct_cloud = float(tslice["eo_cloud_cover"].values)
        ax.set_title(
            f"{lake_name}  {date}\nCloud: {pct_cloud:.1f}%   NaN: {pct_nan:.1f}%",
            color="white",
            fontsize=8,
        )

    # Create animation
    anim = mpl_anim.FuncAnimation(
        fig, frame, frames=n_frames, interval=1000 / fps, blit=False
    )

    # Save
    if outfile.suffix == ".gif":
        anim.save(outfile, writer="imagemagick", fps=fps, dpi=dpi)
    else:
        writer = mpl_anim.FFMpegWriter(fps=fps, metadata={"artist": "sat-tile-stack"})
        anim.save(outfile, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved {outfile} ({n_frames} frames, {fps} fps)")
