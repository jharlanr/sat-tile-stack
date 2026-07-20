"""
Lightweight Flask backend for the labeling GUI.

Serves rendered frames from .nc timestacks and handles label storage.

Usage (console script, after `pip install sat-tile-stack[labeling]`):
    lakelabel --nc_dir path/to/stacks
    lakelabel --nc_dir path/to/stacks --lake_list lakes.csv --labels_csv mine.csv
    Then open http://localhost:5050 in your browser.
"""

import sys
import io
import tempfile
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
import os
import time
from flask import Flask, jsonify, send_file, request, send_from_directory, g

from sat_tile_stack.visualize import _render_frame


# ---------------------------------------------------------------------------
# Module-level config — populated by main()
# ---------------------------------------------------------------------------

NC_DIR = None
LABELS_CSV = None
VAR = "reflectance"
CLASSES = ["ND", "HF", "MD", "LD", "CD"]
LAKE_LIST_IDS = None  # set[str] | None — when set, get_all_ids() filters to this

app = Flask(__name__, static_folder=None)

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

# ---------------------------------------------------------------------------
# On-demand GeoZarr cache
#
# The browser image panel renders each lake's timestack client-side via
# deck.gl-zarr. We convert the (already in-memory) DataArray to a GeoZarr store
# the first time a lake is opened, cache it on disk, and serve it statically.
# Conversion is cheap (~sub-second for a 512^2 tile) and reuses the prefetch
# cache, so it piggybacks on the existing warm-ahead machinery.
# ---------------------------------------------------------------------------

ZARR_CACHE_DIR = Path(tempfile.mkdtemp(prefix="lakelabel_zarr_"))
_zarr_ready = set()
_zarr_lock = threading.Lock()


def ensure_zarr(lake_id):
    """Convert lake_id's timestack to a GeoZarr store if not already cached."""
    with _zarr_lock:
        if lake_id in _zarr_ready:
            return
        out = ZARR_CACHE_DIR / f"{lake_id}.zarr"
        if not (out / "zarr.json").exists():
            from sat_tile_stack.zarr_export import da_to_geozarr
            da_to_geozarr(get_da(lake_id), out)
        _zarr_ready.add(lake_id)


def _load_da_blocking(lake_id):
    """Worker: open .nc and pull the variable fully into memory."""
    nc_path = NC_DIR / f"{lake_id}.nc"
    with xr.open_dataset(nc_path) as ds:
        da = ds[VAR].load()
        # Carry the dataset-level CRS onto the DataArray so the GeoZarr export
        # can geolocate the tile (it's lost when selecting a single variable).
        if "crs" not in da.attrs:
            crs = ds.attrs.get("crs") or ds.attrs.get("epsg")
            if crs is not None:
                da.attrs["crs"] = str(crs)
        return da


def get_da(lake_id):
    """Return DataArray for lake_id, blocking on prefetch if not yet loaded."""
    with _cache_lock:
        if lake_id in _cache:
            return _cache[lake_id]
        fut = _in_flight.get(lake_id)
        if fut is None or fut.cancelled():
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
    if LAKE_LIST_IDS is not None:
        # Preserve the order from --lake_list (the labeler's randomized
        # presentation order). Filter to IDs whose .nc files actually
        # exist in --nc_dir.
        available = {f.stem for f in NC_DIR.glob("*.nc")}
        return [lid for lid in LAKE_LIST_IDS if lid in available]
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


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve the bundled deck.gl viewer widget (and any other static assets)."""
    return send_from_directory(Path(__file__).parent / "static", filename)


@app.route("/api/zarr/<lake_id>/<path:subpath>")
def api_zarr(lake_id, subpath):
    """Serve a file from a lake's on-demand GeoZarr store (zarr.json, chunks)."""
    ensure_zarr(lake_id)
    return send_from_directory(ZARR_CACHE_DIR / f"{lake_id}.zarr", subpath)


# Zarr chunks are stored uncompressed (the browser bundle has no codec wasm),
# so gzip the HTTP responses instead — keeps the widget bundle and the ~1.5 MB
# raw chunks small over the wire (e.g. an SSH tunnel). The browser decompresses
# transparently before zarrita ever sees the bytes.
_GZIP_TYPES = {
    "application/octet-stream",
    "application/javascript",
    "text/javascript",
    "application/json",
    "text/html",
}


@app.before_request
def _req_start():
    if os.environ.get("LAKELABEL_VERBOSE"):
        g._t0 = time.time()


@app.after_request
def _req_log(resp):
    if os.environ.get("LAKELABEL_VERBOSE"):
        dt = (time.time() - getattr(g, "_t0", time.time())) * 1000
        clen = resp.headers.get("Content-Length", "?")
        sys.stderr.write(
            f"REQ {request.method} {request.path} -> {resp.status_code} "
            f"{clen}B {dt:.0f}ms\n"
        )
        sys.stderr.flush()
    return resp


@app.after_request
def _gzip_response(resp):
    import gzip as _gzip
    try:
        if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
            return resp
        if resp.status_code >= 300 or resp.headers.get("Content-Encoding"):
            return resp
        if (resp.content_type or "").split(";")[0] not in _GZIP_TYPES:
            return resp
        resp.direct_passthrough = False
        data = resp.get_data()
        if len(data) < 1024:
            return resp
        comp = _gzip.compress(data, compresslevel=6)
        resp.set_data(comp)
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Vary"] = "Accept-Encoding"
    except Exception:
        return resp
    return resp


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
    # Warm the GeoZarr store so the browser's first zarr.json request is fast.
    ensure_zarr(lake_id)
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
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="lakelabel",
        description="Browser-based labeling GUI for satellite timestacks.",
    )
    parser.add_argument("--nc_dir", type=str, required=True,
                        help="Directory containing .nc timestack files.")
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="Where to save labels (auto-detected from --nc_dir if omitted).")
    parser.add_argument("--lake_list", type=str, default=None,
                        help="CSV with a 'lake_id' column to filter --nc_dir to a specific subset.")
    parser.add_argument("--var", type=str, default="reflectance",
                        help="NetCDF variable name to render (default: reflectance).")
    parser.add_argument("--classes", nargs="+", default=["ND", "HF", "MD", "LD", "CD"],
                        help="Class names (default: ND HF MD LD CD).")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser.")
    args = parser.parse_args(argv)
    if args.labels_csv is None:
        # Auto-detect: if nc_dir is .../CW_2019/stacks, save labels to .../CW_2019/labels_CW_2019.csv
        nc_path = Path(args.nc_dir).resolve()
        parent = nc_path.parent
        parent_name = parent.name
        args.labels_csv = str(parent / f"labels_{parent_name}.csv")
    return args


def _load_lake_list(path):
    """Read a CSV with a 'lake_id' column and return an ordered list of strings.

    The list ordering is preserved so the GUI presents lakes in whatever
    order the CSV specifies (e.g. a per-labeler randomized order for IRR).
    """
    df = pd.read_csv(path)
    if "lake_id" not in df.columns:
        raise ValueError(
            f"--lake_list file {path} must have a 'lake_id' column "
            f"(found columns: {list(df.columns)})"
        )
    return df["lake_id"].astype(str).tolist()


def main(argv=None):
    global NC_DIR, LABELS_CSV, VAR, CLASSES, LAKE_LIST_IDS

    args = parse_args(argv)

    NC_DIR = Path(args.nc_dir)
    LABELS_CSV = Path(args.labels_csv)
    VAR = args.var
    CLASSES = list(args.classes)
    LAKE_LIST_IDS = _load_lake_list(args.lake_list) if args.lake_list else None

    url = f"http://localhost:{args.port}"
    print(f"\nLabeling server")
    print(f"  NC dir:     {NC_DIR}")
    print(f"  Labels CSV: {LABELS_CSV}")
    print(f"  Classes:    {CLASSES}")
    if LAKE_LIST_IDS is not None:
        print(f"  Lake list:  {args.lake_list} ({len(LAKE_LIST_IDS)} ids)")

    if not NC_DIR.exists():
        print(f"\n  ERROR: NC directory does not exist: {NC_DIR}")
        print(f"  Check that the volume is mounted.")
        sys.exit(1)

    n_samples = len(get_all_ids())
    if n_samples == 0:
        if LAKE_LIST_IDS is not None:
            print(f"\n  ERROR: No .nc files in {NC_DIR} match the lake list.")
            print(f"  Check that --nc_dir and --lake_list refer to the same lakes.")
        else:
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

    if not args.no_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import signal

    def shutdown(sig, frame):
        print("\n\n  Labeling server stopped.")
        _prefetch_executor.shutdown(wait=False, cancel_futures=True)
        import shutil as _shutil
        _shutil.rmtree(ZARR_CACHE_DIR, ignore_errors=True)
        print(f"  Labels saved to: {LABELS_CSV}")
        df = load_labels_df()
        n_labeled = len(df.dropna(subset=["label"])) if "label" in df.columns else 0
        print(f"  Total labeled: {n_labeled}/{len(get_all_ids())}")
        print("  Goodbye!\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
