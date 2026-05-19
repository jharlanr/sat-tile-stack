"""Embed expert 5-class drainage labels into the per-lake v2 stacks.

Separate post-build pass (the builds did NOT add labels). Resume-safe and
idempotent: a file that already has `drainage_label` is skipped. No
network, no STAC — just a small in-place CF-1.8 write per file.

Usage:
    python engine/labeling/add_labels_to_stacks.py \
        --stacks_dir <stacks_v2/CW_2018> \
        --labels_csv <labels_CW_2018.csv> [--id_col lake_id] [--workers 8]
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import xarray as xr  # noqa: E402
from sat_tile_stack.coregister import add_labels  # noqa: E402


def _already_labeled(fp):
    try:
        ds = xr.open_dataset(fp)
        try:
            return "drainage_label" in ds.variables
        finally:
            ds.close()
    except Exception:
        return False


def _one(task):
    fp, csv, id_col = task
    lake_id = os.path.splitext(os.path.basename(fp))[0]
    try:
        if _already_labeled(fp):
            return ("skip", lake_id, "")
        add_labels(fp, fp, labels_csv=csv, lake_id=lake_id, id_col=id_col)
        return ("ok", lake_id, "")
    except Exception as e:  # noqa: BLE001
        return ("FAIL", lake_id, repr(e))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stacks_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--id_col", default="lake_id")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.stacks_dir, "*.nc")))
    print(f"{len(files)} .nc in {a.stacks_dir}", flush=True)
    print(f"labels: {a.labels_csv}", flush=True)
    if not files:
        print("ERROR: no .nc files found.")
        return 1
    if not os.path.isfile(a.labels_csv):
        print(f"ERROR: labels CSV not found: {a.labels_csv}")
        return 1

    tasks = [(f, a.labels_csv, a.id_col) for f in files]
    n_ok = n_skip = n_fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for st, lid, msg in ex.map(_one, tasks):
            if st == "ok":
                n_ok += 1
            elif st == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fails.append((lid, msg))
            done = n_ok + n_skip + n_fail
            if done % 100 == 0:
                print(f"  {done}/{len(files)}  "
                      f"(added={n_ok} skip={n_skip} fail={n_fail})",
                      flush=True)

    print(f"\nadded={n_ok} skipped={n_skip} failed={n_fail} "
          f"of {len(files)}")
    for lid, m in fails[:50]:
        print(f"  FAIL {lid}: {m}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
