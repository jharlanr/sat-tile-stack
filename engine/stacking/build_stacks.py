"""
Batch-build .nc timestacks from a lake labels CSV.

Reads a CSV with lake centroids (lon, lat columns) and builds one .nc file
per lake using sattile_stack(). Supports multiprocessing for parallel builds.

Usage:
    # Build all CW 2018 stacks with 32 workers
    python engine/stacking/build_stacks.py \
        --csv data/labels_2018_volumes_CW.csv \
        --output_dir labeling/CW_2018/stacks \
        --id_col new_id \
        --time_range 2018-05-01/2018-09-30 \
        --workers 32

    # Build a subset (rows 0-99)
    python engine/stacking/build_stacks.py \
        --csv data/labels_2018_volumes_CW.csv \
        --output_dir labeling/CW_2018/stacks \
        --id_col new_id \
        --time_range 2018-05-01/2018-09-30 \
        --start 0 --count 100
"""

import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pystac_client
import planetary_computer

from sat_tile_stack import sattile_stack, write_netcdf_from_da


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-build .nc timestacks")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to labels CSV with lon, lat columns")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for stacks")
    parser.add_argument("--id_col", type=str, default="new_id",
                        help="Column name for lake IDs (used as filename)")
    parser.add_argument("--lon_col", type=str, default="lon",
                        help="Column name for longitude")
    parser.add_argument("--lat_col", type=str, default="lat",
                        help="Column name for latitude")
    parser.add_argument("--time_range", type=str, required=True,
                        help="Time range YYYY-MM-DD/YYYY-MM-DD")
    parser.add_argument("--bands", nargs="+",
                        default=["B04", "B03", "B02", "B08", "B11", "SCL"],
                        help="Band names to include")
    parser.add_argument("--collection", type=str, default="sentinel-2-l2a",
                        help="STAC collection ID")
    parser.add_argument("--pix_res", type=int, default=10,
                        help="Pixel resolution in meters")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size in pixels (square)")
    parser.add_argument("--cloudmask", type=str, default="scl",
                        help="Cloud mask method (scl, williamson, or none)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting row index")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of rows to process (None = all)")
    return parser.parse_args()


def build_one_stack(task):
    """Build a single .nc timestack. Designed for multiprocessing.Pool."""
    lake_id, lon, lat, args_dict = task
    outfile = Path(args_dict["output_dir"]) / f"{lake_id}.nc"

    if outfile.exists():
        return {"status": "skip", "id": lake_id}

    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        cloudmask = args_dict["cloudmask"] if args_dict["cloudmask"] != "none" else False

        stack = sattile_stack(
            catalog, (lon, lat),
            band_names=args_dict["bands"],
            collection=args_dict["collection"],
            pix_res=args_dict["pix_res"],
            tile_size=args_dict["tile_size"],
            time_range=args_dict["time_range"],
            normalize=False,
            cloudmask=cloudmask,
            pull_to_mem=True,
        )
        write_netcdf_from_da(stack, str(outfile))
        return {"status": "ok", "id": lake_id}

    except Exception as e:
        return {"status": "error", "id": lake_id, "error": str(e)}


def main():
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.id_col, args.lon_col, args.lat_col])

    # Subset
    end = args.start + args.count if args.count else len(df)
    end = min(end, len(df))
    subset = df.iloc[args.start:end].reset_index(drop=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"sat-tile-stack: Batch Stack Builder")
    print(f"{'='*60}")
    print(f"CSV:        {args.csv} ({len(df)} total lakes)")
    print(f"Building:   {len(subset)} stacks (rows {args.start} to {end-1})")
    print(f"Output:     {args.output_dir}")
    print(f"Time range: {args.time_range}")
    print(f"Tile:       {args.tile_size}x{args.tile_size} @ {args.pix_res}m")
    print(f"Bands:      {args.bands}")
    print(f"Cloud mask: {args.cloudmask}")
    print(f"Workers:    {args.workers}")
    print(f"{'='*60}\n", flush=True)

    # Build task list
    args_dict = {
        "output_dir": str(output_dir),
        "time_range": args.time_range,
        "bands": args.bands,
        "collection": args.collection,
        "pix_res": args.pix_res,
        "tile_size": args.tile_size,
        "cloudmask": args.cloudmask,
    }

    tasks = [
        (row[args.id_col], row[args.lon_col], row[args.lat_col], args_dict)
        for _, row in subset.iterrows()
    ]

    # Run
    start_time = time.time()
    n_ok, n_skip, n_error = 0, 0, 0

    if args.workers == 1:
        for i, task in enumerate(tasks):
            elapsed = time.time() - start_time
            done = n_ok + n_error
            rate = done / elapsed if elapsed > 0 and done > 0 else 0.02
            remaining = (len(tasks) - i) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(tasks)}] {task[0]} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

            result = build_one_stack(task)
            if result["status"] == "ok":
                n_ok += 1
            elif result["status"] == "skip":
                n_skip += 1
            elif result["status"] == "error":
                n_error += 1
                print(f"  ERROR: {result['error']}", flush=True)
    else:
        print(f"Starting {args.workers} parallel workers...\n", flush=True)
        with Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(build_one_stack, tasks):
                total_done = n_ok + n_skip + n_error + 1
                if result["status"] == "ok":
                    n_ok += 1
                    if n_ok % 10 == 0 or total_done == len(tasks):
                        elapsed = time.time() - start_time
                        print(f"  [{total_done}/{len(tasks)}] {n_ok} ok, {n_skip} skip, "
                              f"{n_error} err ({elapsed:.0f}s)", flush=True)
                elif result["status"] == "skip":
                    n_skip += 1
                elif result["status"] == "error":
                    n_error += 1
                    print(f"  ERROR: {result['id']} — {result['error']}", flush=True)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Success: {n_ok}")
    print(f"  Skipped: {n_skip}")
    print(f"  Errors:  {n_error}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
