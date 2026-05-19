"""
CF-1.8 compliance checking for the v2 per-lake stacks.

Two layers:

* ``quick_cf_audit(path)`` — a fast, **dependency-free** structural audit that
  asserts the CF things this pipeline must get right (Conventions, the scalar
  ``crs`` grid-mapping variable + ``grid_mapping`` on every spatial var, x/y/
  time coordinate attrs, flag-variable ``flag_values``/``flag_meanings``
  consistency, ``water_mask_ndwi`` ``_FillValue``). Always available; catches
  the regressions we actually introduce.

* ``check_file`` / ``cf_check`` — the **authoritative** CF checker
  (`cfchecker`, ``cfchecks`` CLI, ``--version 1.8``). Requires the optional
  ``cfchecker`` dependency (``conda install -c conda-forge cfchecker``); the
  structural audit is the fallback when it is not installed.

CLI: ``sts-cf-check <file-or-dir> ...`` (exits 0 iff all clean).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Dependency-free structural audit
# ---------------------------------------------------------------------------

def quick_cf_audit(path):
    """Structural CF-1.8 audit of one file. Returns a list of
    ``(level, message)`` with level in {"ERROR", "WARN"}; empty == clean.

    Opened with ``decode_cf=False`` so on-disk dtypes/attrs (e.g. uint8
    masks, raw ``_FillValue``/``flag_values``) are inspected as written.
    """
    issues = []
    err = lambda m: issues.append(("ERROR", m))
    warn = lambda m: issues.append(("WARN", m))

    ds = xr.open_dataset(path, decode_cf=False)
    try:
        conv = str(ds.attrs.get("Conventions", ""))
        if not conv.startswith("CF-1"):
            err(f"global attr Conventions={conv!r} is not CF-1.x")
        for a in ("title", "source", "history"):
            if not ds.attrs.get(a):
                warn(f"missing recommended global attr {a!r}")

        # grid_mapping variable
        gm_name = "crs"
        if gm_name not in ds.variables:
            err(f"no grid-mapping variable {gm_name!r}")
        else:
            gm = ds[gm_name]
            if not (gm.attrs.get("grid_mapping_name") or gm.attrs.get("crs_wkt")):
                err(f"{gm_name!r} lacks grid_mapping_name/crs_wkt")
            # cfchecker (5.6) does NOT validate crs_wkt syntax — do it here
            # with pyproj so we have positive proof, not an assumption.
            wkt = gm.attrs.get("crs_wkt")
            if wkt:
                try:
                    import pyproj
                    pc = pyproj.CRS.from_wkt(wkt)
                    epsg_attr = gm.attrs.get("epsg_code")
                    if epsg_attr and pc.to_epsg() != int(epsg_attr):
                        err(f"crs_wkt parses to EPSG:{pc.to_epsg()} but "
                            f"epsg_code={epsg_attr}")
                except Exception as e:  # noqa: BLE001
                    err(f"crs_wkt is not valid WKT-CRS: {e}")

        spatial = [v for v in ds.data_vars
                   if {"y", "x"}.issubset(set(ds[v].dims))]
        for v in spatial:
            if ds[v].attrs.get("grid_mapping") != gm_name:
                err(f"{v}: grid_mapping != {gm_name!r}")

        # coordinate attrs
        if "x" in ds.coords:
            ax = ds["x"].attrs
            if ax.get("standard_name") != "projection_x_coordinate":
                err("x: standard_name != projection_x_coordinate")
            if not ax.get("units"):
                err("x: missing units")
            if ax.get("axis") != "X":
                warn("x: axis != 'X'")
        if "y" in ds.coords:
            ay = ds["y"].attrs
            if ay.get("standard_name") != "projection_y_coordinate":
                err("y: standard_name != projection_y_coordinate")
            if not ay.get("units"):
                err("y: missing units")
        if "time" in ds.coords:
            tu = ds["time"].attrs.get("units", "")
            if "since" not in tu:
                err(f"time: non-CF units {tu!r}")
            if not ds["time"].attrs.get("calendar"):
                warn("time: missing calendar")

        # flag variables: flag_values dtype must match var dtype; counts align
        for v in ds.data_vars:
            fv = ds[v].attrs.get("flag_values")
            fm = ds[v].attrs.get("flag_meanings")
            if fv is None and fm is None:
                continue
            if fv is None or fm is None:
                err(f"{v}: only one of flag_values/flag_meanings present")
                continue
            fv = np.atleast_1d(fv)
            if fv.dtype != ds[v].dtype:
                err(f"{v}: flag_values dtype {fv.dtype} != var dtype "
                    f"{ds[v].dtype}")
            if len(str(fm).split()) != fv.size:
                err(f"{v}: {fv.size} flag_values but "
                    f"{len(str(fm).split())} flag_meanings")

        if "water_mask_ndwi" in ds.variables:
            wm = ds["water_mask_ndwi"]
            fillv = wm.attrs.get("_FillValue", wm.encoding.get("_FillValue"))
            if fillv is None:
                err("water_mask_ndwi: no _FillValue (missing-data sentinel)")
            if str(wm.dtype) != "uint8":
                warn(f"water_mask_ndwi dtype {wm.dtype} (expected uint8)")
    finally:
        ds.close()
    return issues


# ---------------------------------------------------------------------------
# Authoritative cfchecker wrapper
# ---------------------------------------------------------------------------

def _cfchecks_exe():
    return shutil.which("cfchecks")


def check_file(path, version="1.8"):
    """Run the authoritative ``cfchecks`` on one file.

    Returns ``{path, n_errors, n_warnings, ok, output}``. Raises
    ``RuntimeError`` if ``cfchecker`` is not installed.
    """
    exe = _cfchecks_exe()
    if exe is None:
        raise RuntimeError(
            "cfchecker not installed — `conda install -c conda-forge "
            "cfchecker` (or pip install cfchecker; needs UDUNITS2)."
        )
    res = subprocess.run([exe, "-v", str(version), str(path)],
                          capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    me = re.search(r"ERRORS detected:\s*(\d+)", out)
    mw = re.search(r"WARNINGS given:\s*(\d+)", out)
    n_err = int(me.group(1)) if me else None
    n_warn = int(mw.group(1)) if mw else None
    ok = (n_err == 0) if n_err is not None else (res.returncode == 0)
    return {"path": str(path), "n_errors": n_err, "n_warnings": n_warn,
            "ok": ok, "output": out}


def _expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.nc"),
                                      recursive=True))
        else:
            files.append(p)
    return files


def cf_check(paths, version="1.8", workers=8, quick=True):
    """Validate files/dirs. Always runs the structural audit; additionally
    runs ``cfchecker`` when available.

    Returns ``(all_ok, results)`` where each result has ``path``,
    ``quick_errors``/``quick_warnings`` and (if cfchecker ran) ``cf_errors``.
    """
    files = _expand(list(paths))
    have_cfchecks = _cfchecks_exe() is not None
    results = []

    def _one(f):
        r = {"path": f}
        if quick:
            qi = quick_cf_audit(f)
            r["quick_errors"] = [m for lvl, m in qi if lvl == "ERROR"]
            r["quick_warnings"] = [m for lvl, m in qi if lvl == "WARN"]
        if have_cfchecks:
            try:
                cr = check_file(f, version=version)
                r["cf_errors"] = cr["n_errors"]
                r["cf_warnings"] = cr["n_warnings"]
                r["cf_output"] = cr["output"]
            except Exception as e:  # noqa: BLE001
                r["cf_error_msg"] = str(e)
        return r

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(_one, files))

    all_ok = True
    for r in results:
        ok = not r.get("quick_errors")
        if "cf_errors" in r and r["cf_errors"]:
            ok = False
        all_ok = all_ok and ok
    return all_ok, results, have_cfchecks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="CF-1.8 check (structural audit + cfchecker if installed)")
    p.add_argument("paths", nargs="+", help=".nc files or directories")
    p.add_argument("-v", "--version", default="1.8")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-quick", action="store_true",
                   help="skip the dependency-free structural audit")
    a = p.parse_args(argv)

    all_ok, results, have = cf_check(
        a.paths, version=a.version, workers=a.workers, quick=not a.no_quick)
    status = ("AVAILABLE" if have
              else "NOT installed (structural pre-check only)")
    print(f"standard cfchecker: {status}\n")
    for r in sorted(results, key=lambda x: x["path"]):
        qn = len(r.get("quick_errors", []))
        cn = r.get("cf_errors")
        tag = "PASS" if (qn == 0 and not cn) else "FAIL"
        extra = f" cf_err={cn}" if cn is not None else (
            f" ({r['cf_error_msg']})" if "cf_error_msg" in r else "")
        print(f"[{tag}] {r['path']}  quick_err={qn} "
              f"quick_warn={len(r.get('quick_warnings', []))}{extra}")
        for m in r.get("quick_errors", []):
            print(f"        ERROR {m}")
    print(f"\n{'ALL CLEAN' if all_ok else 'FAILURES PRESENT'} "
          f"({len(results)} file(s))")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
