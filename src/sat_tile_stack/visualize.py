import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from pathlib import Path

def timestack_to_movie(
        nc_path,
        var="reflectance",               # data variable name in the .nc file
        bands=("B04","B03","B02"),       # tuple of band labels for RGB (None → grayscale)
        stretch=(2, 98),                 # percentile stretch per frame
        fps=6,                           # frames-per-second
        outfile=None,                    # "movie.mp4" or "movie.gif"
        dpi=150                          # output resolution
    ):
    """
    Build a quicklook movie from a time×band×y×x NetCDF stack.

    Parameters
    ----------
    nc_path : str | Path
        Path to the NetCDF file.
    var : str
        Name of the 4-D data variable (time, band, y, x).
    bands : 3-tuple of str | None
        Which band labels to use for RGB.  Use `None` for single-band grayscale.
    stretch : (low, high)
        Percentiles for linear contrast stretch per frame.
    fps : int
        Frames per second for the output video.
    outfile : str | Path | None
        Output filename.  Inferred from extension:
            *.mp4 → H.264 (needs ffmpeg)
            *.gif → animated GIF
        If None, uses the NetCDF name + ".mp4".
    dpi : int
        DPI passed to the animation writer.
    """
    nc_path = Path(nc_path)
    if outfile is None:
        outfile = nc_path.with_suffix(".mp4")
    outfile = Path(outfile)

    # --- load lazily (only one frame at a time) ----------------------------
    ds = xr.open_dataset(nc_path)
    data = ds[var]                       # (time, band, y, x)

    times = data.time.values
    n_frames = len(times)

    # --- setup figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    img_artist = ax.imshow(np.zeros((data.sizes["y"], data.sizes["x"])),
                           cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    title = ax.set_title("")

    def linear_stretch(arr, pmin, pmax):
        lo, hi = np.nanpercentile(arr, (pmin, pmax))
        return np.clip((arr - lo) / (hi - lo), 0, 1)

    def frame(i):
        da = data.isel(time=i).load()    # load one timestep

        if bands is None:               # single-band
            arr = linear_stretch(da.values[0], *stretch)
            rgb = arr
            img_artist.set_cmap("gray")
        else:                           # RGB compose
            rgb = [da.sel(band=b).values for b in bands]
            rgb = np.stack(rgb, axis=-1)             # y×x×3
            # rgb = linear_stretch(rgb, *stretch)
            rgb = rgb/10000
        img_artist.set_data(rgb)
        title.set_text(str(np.datetime_as_string(times[i], unit="D")))
        return img_artist, title

    anim = mpl_anim.FuncAnimation(fig, frame, frames=n_frames, blit=True)

    # --- choose writer based on extension ----------------------------------
    if outfile.suffix == ".gif":
        anim.save(outfile, writer="imagemagick", fps=fps, dpi=dpi)
    else:                               # default: .mp4
        Writer = mpl_anim.writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={"artist": "timestack"})   # ← no dpi here
        anim.save(outfile, writer=writer, dpi=dpi)                   # dpi belongs here

    plt.close(fig)
    print(f"📽  saved {outfile} ({n_frames} frames, {fps} fps)")

