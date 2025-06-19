import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from pathlib import Path

def timestack_to_movie(
        nc_path,
        var="reflectance",
        bands=("B04","B03","B02"),
        fps=6,
        outfile=None,
        dpi=150
    ):
    """
    Build a quicklook movie from a time×band×y×x NetCDF stack,
    normalizing pixels by 10000, forcing a black background,
    and outlining the mask (last band) on each frame.
    """
    nc_path = Path(nc_path)
    outfile = Path(outfile or nc_path.with_suffix(".mp4"))

    ds   = xr.open_dataset(nc_path)
    data = ds[var]            # (time, band, y, x)
    times = data.time.values
    n_frames = len(times)

    # set up the figure
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')

    def frame(i):
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')

        # load one timestep
        tslice = data.isel(time=i).load()

        # build RGB exactly like plot_n_days
        r = tslice.isel(band=0).values / 10000.0
        g = tslice.isel(band=1).values / 10000.0
        b = tslice.isel(band=2).values / 10000.0
        rgb = np.clip(np.stack([r,g,b], axis=-1), 0, 1)
        ax.imshow(rgb)

        # contour the mask (last band)
        mask2d = tslice.isel(band=-1).values
        ax.contour(mask2d, levels=[0.5], colors="red", linewidths=1)

        # title with lake name, date, clouds, nans
        date = np.datetime_as_string(times[i], unit='D')
        lake_name = nc_path.stem.replace("tstack_", "")
        pct_nan   = float(tslice["pct_nans"].values)
        pct_cloud = float(tslice["eo_cloud_cover"].values)
        ax.set_title(
            f"{lake_name}  {date}\nCloud: {pct_cloud:.1f}%   NaN: {pct_nan:.1f}%",
            color='white',
            fontsize=8
        )

    # note: interval in ms = 1000/fps, blit=False so we can clear+redraw
    anim = mpl_anim.FuncAnimation(
        fig, frame,
        frames=n_frames,
        interval=1000/fps,
        blit=False
    )

    # save
    if outfile.suffix == ".gif":
        anim.save(outfile, writer="imagemagick", fps=fps, dpi=dpi)
    else:
        writer = mpl_anim.FFMpegWriter(fps=fps, metadata={'artist':'timestack'})
        anim.save(outfile, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"📽  saved {outfile} ({n_frames} frames, {fps} fps)")
    
    

# def timestack_to_movie(
#         nc_path,
#         var="reflectance",
#         bands=("B04","B03","B02"),
#         fps=6,
#         outfile=None,
#         dpi=150
#     ):
#     """
#     Build a quicklook movie from a time×band×y×x NetCDF stack,
#     normalizing pixels by 10000, and forcing a black background.
#     """
#     nc_path = Path(nc_path)
#     if outfile is None:
#         outfile = nc_path.with_suffix(".mp4")
#     outfile = Path(outfile)

#     # --- load lazily (only one frame at a time) ----------------------------
#     ds = xr.open_dataset(nc_path)
#     data = ds[var]                       # (time, band, y, x)

#     times = data.time.values
#     n_frames = len(times)

#     # --- setup figure ------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # <--- Make figure & axes background black:
#     fig.patch.set_facecolor('black')
#     ax.set_facecolor('black')

#     # Create an initial "blank" image.  Since we will feed in RGB values in [0,1],
#     # a zero array will appear as black.
#     img_artist = ax.imshow(
#         np.zeros((data.sizes["y"], data.sizes["x"], 3)), 
#         vmin=0, vmax=1
#     )
#     ax.axis("off")

#     # Initialize a white title (so it shows up on black).  We'll overwrite its text each frame.
#     title = ax.set_title("", color='white')

#     def frame(i):
#         da = data.isel(time=i).load()    # load one timestep

#         if bands is None:  # single-band (grayscale)
#             arr = da.values[0] / 10000.0
#             arr = np.clip(arr, 0, 1)
#             rgb = np.stack([arr]*3, axis=-1)  # replicate to RGB channels
#         else:
#             # Pull each band, divide by 10000, clip to [0,1], then stack as RGB
#             rgb_bands = []
#             for b in bands:
#                 band_arr = da.sel(band=b).values / 10000.0
#                 band_arr = np.clip(band_arr, 0, 1)
#                 rgb_bands.append(band_arr)
#             rgb = np.stack(rgb_bands, axis=-1)  # shape (y, x, 3)

#         img_artist.set_data(rgb)

#         # Put "lakenum" + date at the top, in white text
#         date_str = np.datetime_as_string(times[i], unit="D")
#         lake_name = nc_path.stem.replace("tstack_", "")
#         title.set_text(f"{lake_name}   {date_str}")
#         # Title color was already set to white above, so no need to reset it here.

#         return img_artist, title

#     anim = mpl_anim.FuncAnimation(fig, frame, frames=n_frames, blit=True)

#     # --- choose writer based on extension ----------------------------------
#     if outfile.suffix == ".gif":
#         anim.save(outfile, writer="imagemagick", fps=fps, dpi=dpi)
#     else:
#         Writer = mpl_anim.writers["ffmpeg"]
#         writer = Writer(fps=fps, metadata={"artist": "timestack"})
#         anim.save(outfile, writer=writer, dpi=dpi)

#     plt.close(fig)
#     print(f"📽  saved {outfile} ({n_frames} frames, {fps} fps)")

