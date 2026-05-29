# sat-tile-stack demos

Two notebooks plus one example NetCDF, runnable from the command line.

- [`demo_essd_sattilestack.ipynb`](demo_essd_sattilestack.ipynb) — builds a daily Sentinel-2 RGB time-stack over the Port of Oakland using `sat_tile_stack.sattile_stack(...)` and plots a few frames. The shortest possible introduction to the package; needs an internet connection to hit the Microsoft Planetary Computer STAC.
- [`demo_essd_readdata.ipynb`](demo_essd_readdata.ipynb) — opens [`CW2019_1907.nc`](CW2019_1907.nc) (one lake from the [ESSD companion dataset](https://doi.org/10.25740/sf350xp4038), bundled here, ~170 MB) and walks through the full v2 schema: imagery → reflectance conversion, embedded drainage-mechanism label, static lake boundary, dynamic NDWI water mask, `p_water` scalar series. Fully offline.

## Setup

```bash
# 1. Clone the repo and cd in
git clone https://github.com/jharlanr/sat-tile-stack.git
cd sat-tile-stack

# 2. Create and activate the conda env (one-time)
conda env create -f environment.yml
conda activate sat-tile-stack

# 3. Launch JupyterLab in the demos directory
jupyter lab demos/
```

The `environment.yml` recipe installs the package itself (`pip install -e .[dev]`), JupyterLab, and the non-Python tools (`ffmpeg`, `cfchecker`) — so you can run both notebooks straight after the activate step.

## If you only want a minimal pip install

```bash
git clone https://github.com/jharlanr/sat-tile-stack.git
cd sat-tile-stack
pip install -e .
pip install jupyterlab
jupyter lab demos/
```

## Run order

Either notebook works on its own; there's no shared state between them.

- **Open the package demo first** if you're new to `sat-tile-stack` and want to see how an AOI-centered time-stack is built from STAC.
- **Open the read-data demo first** if you've downloaded a per-lake `.nc` from the SDR deposit and want to confirm everything's intact and understand the schema before training a model.
