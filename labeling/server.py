"""
Lightweight Flask backend for the labeling GUI.

Serves rendered frames from .nc timestacks and handles label storage.

Usage:
    python labeling/server.py --nc_dir path/to/stacks
    Then open http://localhost:5050 in your browser.
"""

import sys
import io
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['figure.max_open_warning'] = 0  # suppress warning
import matplotlib.pyplot as plt
from flask import Flask, jsonify, send_file, request, send_from_directory

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sat_tile_stack.visualize import _render_frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Labeling server")
    parser.add_argument("--nc_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--var", type=str, default="reflectance")
    parser.add_argument("--classes", nargs="+", default=["ND", "HF", "MD", "LD", "CD"])
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    if args.labels_csv is None:
        # Auto-detect: if nc_dir is .../CW_2019/stacks, save labels to .../CW_2019/labels_CW_2019.csv
        nc_path = Path(args.nc_dir).resolve()
        parent = nc_path.parent  # e.g., .../CW_2019
        parent_name = parent.name  # e.g., CW_2019
        args.labels_csv = str(parent / f"labels_{parent_name}.csv")
    return args


args = parse_args()
app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

NC_DIR = Path(args.nc_dir)
LABELS_CSV = Path(args.labels_csv)
VAR = args.var
CLASSES = args.classes

# ---------------------------------------------------------------------------
# Windowed prefetch cache
#
# Holds at most (PREFETCH_AHEAD + 1) fully-loaded DataArrays — the current
# sample plus the next N unlabeled samples. A single background worker
# pre-loads upcoming samples while the user labels the current one, so
# /api/info for the next lake hits a warm cache instead of a 60s cold read.
#
# Memory is hard-capped: every time the user advances, _slide_window evicts
# anything outside [current, current+PREFETCH_AHEAD]. Memory does NOT
# compound over a labeling session.
# ---------------------------------------------------------------------------

PREFETCH_AHEAD = 2  # cache size = PREFETCH_AHEAD + 1; ~600MB peak at 200MB/file

_cache = {}            # lake_id -> fully-loaded DataArray
_in_flight = {}        # lake_id -> Future
_cache_lock = threading.Lock()
_prefetch_executor = ThreadPoolExecutor(max_workers=1)


def _load_da_blocking(lake_id):
    """Worker: open .nc and pull the variable fully into memory."""
    nc_path = NC_DIR / f"{lake_id}.nc"
    with xr.open_dataset(nc_path) as ds:
        return ds[VAR].load()


def get_da(lake_id):
    """Return DataArray for lake_id, blocking on prefetch if not yet loaded."""
    with _cache_lock:
        if lake_id in _cache:
            return _cache[lake_id]
        fut = _in_flight.get(lake_id)
        if fut is None:
            fut = _prefetch_executor.submit(_load_da_blocking, lake_id)
            _in_flight[lake_id] = fut
    da = fut.result()
    with _cache_lock:
        _cache[lake_id] = da
        _in_flight.pop(lake_id, None)
    return da


def _unlabeled_ids():
    all_ids = get_all_ids()
    df = load_labels_df()
    if "label" in df.columns:
        labeled = set(df.dropna(subset=["label"])["lake_id"].astype(str))
        labeled |= set(df[df["label"].astype(str) != ""]["lake_id"].astype(str))
    else:
        labeled = set()
    return [fid for fid in all_ids if fid not in labeled]


def slide_window(current_id):
    """Evict everything outside [current, current+PREFETCH_AHEAD] in the
    unlabeled-ID order, then schedule prefetches for any window slot not
    already cached or in flight.
    """
    unlabeled = _unlabeled_ids()
    try:
        idx = unlabeled.index(current_id)
        window = unlabeled[idx:idx + PREFETCH_AHEAD + 1]
    except ValueError:
        # current_id is already labeled (e.g. user revisiting); just pin it
        window = [current_id]

    keep = set(window)
    with _cache_lock:
        for lid in list(_cache.keys()):
            if lid not in keep:
                del _cache[lid]
        for lid in list(_in_flight.keys()):
            if lid not in keep:
                fut = _in_flight.pop(lid)
                fut.cancel()
        for lid in keep:
            if lid not in _cache and lid not in _in_flight:
                _in_flight[lid] = _prefetch_executor.submit(
                    _load_da_blocking, lid
                )


def get_all_ids():
    return sorted([f.stem for f in NC_DIR.glob("*.nc")])


def load_labels_df():
    prob_cols = [f"p_{cn}" for cn in CLASSES]
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        # Ensure correct dtypes to avoid FutureWarning
        if "flagged" in df.columns:
            df["flagged"] = df["flagged"].fillna(False).astype(bool)
        if "notes" in df.columns:
            df["notes"] = df["notes"].fillna("").astype(str)
        if "label" in df.columns:
            df["label"] = df["label"].fillna("").astype(str)
        if "lake_id" in df.columns:
            df["lake_id"] = df["lake_id"].fillna("").astype(str)
        return df
    else:
        LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=["lake_id", "label"] + prob_cols + ["notes", "flagged"])


def save_labels_df(df):
    df = df.sort_values("lake_id", key=lambda s: s.str.extract(r"(\d+)$")[0].astype(int)).reset_index(drop=True)
    df.to_csv(LABELS_CSV, index=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/samples")
def api_samples():
    """List all samples with labeled/unlabeled status."""
    all_ids = get_all_ids()
    df = load_labels_df()
    labeled = set()
    flagged = set()
    if "label" in df.columns:
        labeled = set(df.dropna(subset=["label"])["lake_id"].astype(str))
    if "flagged" in df.columns:
        flagged = set(df[df["flagged"] == True]["lake_id"].astype(str))

    samples = []
    for fid in all_ids:
        samples.append({
            "id": fid,
            "labeled": fid in labeled,
            "flagged": fid in flagged,
        })
    return jsonify(samples)


@app.route("/api/info/<lake_id>")
def api_info(lake_id):
    """Get metadata for a sample."""
    da = get_da(lake_id)
    # Slide the prefetch window so upcoming samples warm in the background
    # while the user labels this one.
    slide_window(lake_id)
    dates = [np.datetime_as_string(t, unit="D") for t in da.time.values]
    bands = [str(b) for b in da.band.values]
    return jsonify({
        "id": lake_id,
        "n_frames": len(dates),
        "dates": dates,
        "bands": bands,
        "shape": list(da.shape),
    })


@app.route("/api/label/<lake_id>")
def api_get_label(lake_id):
    """Get existing label for a sample (if any)."""
    df = load_labels_df()
    if lake_id in df["lake_id"].values:
        row = df[df["lake_id"] == lake_id].iloc[0]
        result = {"lake_id": lake_id, "labeled": pd.notna(row.get("label"))}
        if result["labeled"]:
            result["label"] = str(row["label"])
            result["notes"] = str(row.get("notes", "")) if pd.notna(row.get("notes")) else ""
            result["flagged"] = bool(row.get("flagged", False))
            result["probs"] = {}
            for cn in CLASSES:
                col = f"p_{cn}"
                result["probs"][cn] = float(row[col]) if col in row and pd.notna(row[col]) else 0.0
        return jsonify(result)
    return jsonify({"lake_id": lake_id, "labeled": False})


@app.route("/api/frame/<lake_id>/<int:frame_idx>")
def api_frame(lake_id, frame_idx):
    """Render a frame as PNG."""
    da = get_da(lake_id)
    bands_available = [str(b) for b in da.band.values]

    # Pick RGB bands
    rgb_options = [
        ["B04", "B03", "B02"],
        ["SR_B4", "SR_B3", "SR_B2"],
    ]
    rgb_bands = None
    for opt in rgb_options:
        if all(b in bands_available for b in opt):
            rgb_bands = opt
            break
    if rgb_bands is None:
        rgb_bands = bands_available[:3]

    tslice = da.isel(time=frame_idx)
    band_data = tslice.sel(band=rgb_bands)

    plt.close('all')  # prevent memory leak from rapid requests
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if np.isnan(band_data.values).all():
        pass  # just show black frame
    else:
        rgb = _render_frame(tslice, rgb_bands, "divide", {})
        ax.imshow(rgb)
        if "mask" in bands_available:
            mask = tslice.sel(band="mask").values
            if not np.isnan(mask).all():
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/ping")
def api_ping():
    """Lightweight heartbeat — no disk I/O."""
    return jsonify({"status": "ok"})


@app.route("/api/progress")
def api_progress():
    """Get labeling progress and class distribution."""
    all_ids = get_all_ids()
    df = load_labels_df()

    labeled = set()
    if "label" in df.columns:
        labeled = set(df.dropna(subset=["label"])["lake_id"].astype(str))

    counts = {}
    if "label" in df.columns:
        counts = dict(Counter(df.dropna(subset=["label"])["label"].astype(str)))

    return jsonify({
        "total": len(all_ids),
        "labeled": len(labeled),
        "remaining": len(all_ids) - len(labeled),
        "classes": CLASSES,
        "counts": counts,
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    """Save a label."""
    data = request.json
    lake_id = data["lake_id"]
    probs = data["probs"]  # dict: {class_name: probability}
    notes = data.get("notes", "")
    flagged = data.get("flagged", False)

    # Compute argmax
    max_prob = max(probs.values())
    label = max(probs, key=probs.get) if max_prob > 0 else ""

    df = load_labels_df()
    row = {"lake_id": lake_id, "label": label, "notes": notes, "flagged": flagged}
    for cn in CLASSES:
        row[f"p_{cn}"] = probs.get(cn, 0)

    if lake_id in df["lake_id"].values:
        idx = df[df["lake_id"] == lake_id].index[0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    save_labels_df(df)

    return jsonify({"status": "ok", "label": label, "lake_id": lake_id})


@app.route("/api/flag", methods=["POST"])
def api_flag():
    """Toggle flag for revisit."""
    data = request.json
    lake_id = data["lake_id"]

    df = load_labels_df()
    if lake_id in df["lake_id"].values:
        idx = df[df["lake_id"] == lake_id].index[0]
        current = df.at[idx, "flagged"] if "flagged" in df.columns else False
        new_flag = not current
        df.at[idx, "flagged"] = new_flag
    else:
        new_flag = True
        row = {"lake_id": lake_id, "flagged": True}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    save_labels_df(df)
    return jsonify({"status": "ok", "flagged": new_flag})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser

    url = f"http://localhost:{args.port}"
    print(f"\nLabeling server")
    print(f"  NC dir:     {NC_DIR}")
    print(f"  Labels CSV: {LABELS_CSV}")
    print(f"  Classes:    {CLASSES}")

    if not NC_DIR.exists():
        print(f"\n  ERROR: NC directory does not exist: {NC_DIR}")
        print(f"  Check that the volume is mounted.")
        sys.exit(1)

    n_samples = len(get_all_ids())
    if n_samples == 0:
        print(f"\n  ERROR: No .nc files found in {NC_DIR}")
        print(f"  Directory exists but contains no NetCDF files. Check the mount.")
        sys.exit(1)

    print(f"  Samples:    {n_samples}")
    print(f"  Prefetch:   {PREFETCH_AHEAD} ahead (cache size {PREFETCH_AHEAD + 1})")

    # Pre-warm: start loading the first unlabeled sample now so the browser's
    # initial /api/info request hits a (partially) warm cache.
    _initial = _unlabeled_ids()
    if _initial:
        slide_window(_initial[0])

    print(f"\n  Opening {url} ...\n")

    # Silence Flask request logging
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # Open browser after a short delay (so Flask is ready)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import signal

    def shutdown(sig, frame):
        print("\n\n  Labeling server stopped.")
        _prefetch_executor.shutdown(wait=False, cancel_futures=True)
        print(f"  Labels saved to: {LABELS_CSV}")
        df = load_labels_df()
        n_labeled = len(df.dropna(subset=["label"])) if "label" in df.columns else 0
        print(f"  Total labeled: {n_labeled}/{len(get_all_ids())}")
        print("  Goodbye!\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    app.run(port=args.port, debug=False)
