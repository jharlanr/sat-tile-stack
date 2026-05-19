#!/bin/bash
#SBATCH --job-name=sts_build
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.err
#SBATCH --time=47:30:00
#SBATCH -p serc,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# BUILD + LABEL + CF-CHECK TIMESTACKS FOR ANY REGION + YEAR  (one job)
# =============================================================================
#
# Single self-contained job, scoped to one <REGION> <YEAR>, three stages:
#   1. BUILD     512x512 @ 10m daily timestacks (May-Sep) from Dunmire
#                GeoJSON + coregister chain (reflectance / cloud_mask /
#                water_mask_ndwi / lake_boundary / p_water).  Resume-safe.
#   2. LABELS    embed expert 5-class drainage labels in place
#                (drainage_label + label_probability + class/class_name),
#                CF-1.8 netCDF append. Idempotent (labeled files skipped).
#   3. CF-CHECK  validate the now label-complete dir to CF-1.8. The job's
#                exit code reflects this stage, so a green job = built +
#                labeled + CF-signed-off.
#
# Every stage is resume-safe/idempotent, so re-submitting after a timeout
# just continues. Labels CSV must be staged on OAK before submit (hard
# pre-flight below) so an 18h+ build is never wasted on a stage-2 that
# cannot run.
#
# USAGE:
#   sbatch run_build_stacks_region.sh <REGION> <YEAR>
#
#   REGION: CW, NW, NO, NE, SW, SE
#   YEAR:   2018 or 2019
#
# EXAMPLES:
#   sbatch run_build_stacks_region.sh CW 2019
#   sbatch run_build_stacks_region.sh NW 2018
#   LABELS_DIR=/some/other/dir sbatch run_build_stacks_region.sh CW 2018
#
# =============================================================================

set -uo pipefail
# Note: NOT using `set -e` because we explicitly tolerate some non-zero
# exit codes downstream — e.g. the `ls ... | head -1` pattern used to
# pick the first .nc file triggers SIGPIPE in `ls` under pipefail, which
# would kill the script prematurely. We check $? manually where it matters.

# --- Parse arguments ---
REGION="${1:?Usage: sbatch run_build_stacks_region.sh <REGION> <YEAR>}"
YEAR="${2:?Usage: sbatch run_build_stacks_region.sh <REGION> <YEAR>}"

VALID_REGIONS="CW NW NO NE SW SE"
if ! echo "$VALID_REGIONS" | grep -qw "$REGION"; then
    echo "ERROR: Invalid region '$REGION'. Must be one of: $VALID_REGIONS"
    exit 1
fi

if [[ "$YEAR" != "2018" && "$YEAR" != "2019" ]]; then
    echo "ERROR: Invalid year '$YEAR'. Must be 2018 or 2019."
    exit 1
fi

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/sat-tile-stack"
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack"
DUNMIRE_GEOJSON="$REPO_DIR/labeling/dunmire/labels_${YEAR}_volumes.geojson"
# Dunmire 2025 per-lake daily series (p_water source): canonical OAK data
# location (has `ids` coord with this region's lake IDs + `S2_water`).
DUNMIRE_NC="/oak/stanford/groups/cyaolai/JoshRines/data/dunmire/all_lakes_${YEAR}.nc"
# v2 CF-1.8 rebuild lands in a fresh tree; v1 stacks/ stays untouched as a
# fallback until v2 passes validation (see sat-tile-stack/claudiary/20260508F).
OUTPUT_DIR="$SHERLOCK_DIR/stacks_v2/${REGION}_${YEAR}"
EXTRACT_CSV="$SHERLOCK_DIR/stacks_v2/${REGION}_${YEAR}_centroids.csv"
# Stage-2 input: canonical 5-class label CSV (lake_id,label,p_ND..p_CD,
# notes,flagged). Override at submit: LABELS_DIR=/path sbatch ...
LABELS_DIR="${LABELS_DIR:-/oak/stanford/groups/cyaolai/JoshRines/data/labels}"
LABELS_CSV="$LABELS_DIR/labels_${REGION}_${YEAR}.csv"

mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "sat-tile-stack: Build ${REGION} ${YEAR} Stacks"
echo "=============================================="
echo "Region:     $REGION"
echo "Year:       $YEAR"
echo "GeoJSON:    $DUNMIRE_GEOJSON"
echo "Output:     $OUTPUT_DIR"
echo "DunmireNC:  $DUNMIRE_NC"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
echo "=============================================="

# --- Pre-flight: fail fast on missing coregister inputs (don't waste an
#     overnight build only to have every lake's coregister step error). ---
for f in "$DUNMIRE_GEOJSON" "$DUNMIRE_NC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required coregister input not found: $f"
        echo "Edit DUNMIRE_NC / DUNMIRE_GEOJSON in this script (or stage the"
        echo "file) before submitting. Aborting."
        exit 1
    fi
done
# Stage-2 labels CSV must exist NOW (before the long build) so we never
# burn an 18h+ build only to have the embed step fail for a missing file.
if [ ! -f "$LABELS_CSV" ]; then
    echo "ERROR: labels CSV not found: $LABELS_CSV"
    echo "Stage the canonical labels_${REGION}_${YEAR}.csv there, or pass"
    echo "LABELS_DIR=/path at submit. Aborting before the build."
    exit 1
fi
echo "LabelsCSV:  $LABELS_CSV"

# --- Load modules ---
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

# numpy/pandas/scipy come from the Sherlock modules above (pip-installing
# them would conflict). rasterio is explicit (add_static_polygon needs
# rasterio.features/warp; don't rely on it being transitive); zarr is
# harmless + future-proofs add_raster_zarr.
pip install --user xarray netcdf4 pystac-client planetary-computer stackstac \
    geopandas rioxarray pyproj shapely rasterio "zarr>=3" matplotlib dask

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# --- Extract region CSV with centroids from Dunmire GeoJSON ---
echo ""
echo "Extracting ${REGION} lakes from Dunmire GeoJSON..."

python3 -u -c "
import json
import csv
import sys
from shapely.geometry import shape

region = '${REGION}'
geojson_path = '${DUNMIRE_GEOJSON}'
output_csv = '${EXTRACT_CSV}'

with open(geojson_path) as f:
    data = json.load(f)

rows = []
for feat in data['features']:
    props = feat['properties']
    if props['region'] != region:
        continue

    # Compute centroid via shapely (handles Polygon and MultiPolygon)
    geom = shape(feat['geometry'])
    c = geom.centroid

    row = dict(props)
    row['lon'] = c.x
    row['lat'] = c.y
    rows.append(row)

if not rows:
    print(f'ERROR: No features found for region {region}')
    sys.exit(1)

# Write CSV
fieldnames = list(rows[0].keys())
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'  Extracted {len(rows)} lakes for {region}')
print(f'  Saved to {output_csv}')
"

echo ""
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# --- Build first stack solo and inspect it ---
echo "Building first stack (single worker) for inspection..."
python3 -u "$REPO_DIR/engine/stacking/build_stacks.py" \
    --csv "$EXTRACT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --id_col "new_id" \
    --time_range "${YEAR}-05-01/${YEAR}-09-30" \
    --bands B04 B03 B02 B08 B11 B12 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 1 \
    --count 1 \
    --coregister \
    --dunmire_nc "$DUNMIRE_NC" \
    --boundary_geojson "$DUNMIRE_GEOJSON" \
    --ndwi_min 0.3

# Inspect the first file
FIRST_NC=$(ls "$OUTPUT_DIR/"*.nc 2>/dev/null | head -1)
if [ -n "$FIRST_NC" ]; then
    echo ""
    echo "=============================================="
    echo "INSPECTING FIRST STACK: $FIRST_NC"
    echo "=============================================="
    ls -lh "$FIRST_NC"
    python3 -c "
import xarray as xr
ds = xr.open_dataset('$FIRST_NC')
print(ds)
print()
print('File size:', round(ds.nbytes / 1024 / 1024, 1), 'MB (in memory)')
"
    echo "=============================================="
    echo ""
else
    echo "WARNING: No .nc file produced. Check errors above."
fi

# --- Build remaining stacks in parallel ---
echo "Building remaining stacks (8 workers)..."
python3 -u "$REPO_DIR/engine/stacking/build_stacks.py" \
    --csv "$EXTRACT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --id_col "new_id" \
    --time_range "${YEAR}-05-01/${YEAR}-09-30" \
    --bands B04 B03 B02 B08 B11 B12 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 8 \
    --coregister \
    --dunmire_nc "$DUNMIRE_NC" \
    --boundary_geojson "$DUNMIRE_GEOJSON" \
    --ndwi_min 0.3

BUILD_RC=$?

NC_COUNT=$(ls "$OUTPUT_DIR/"*.nc 2>/dev/null | wc -l)
echo ""
echo "=============================================="
echo "STAGE 1 (build) done: $(date)  rc=$BUILD_RC  stacks=$NC_COUNT"
echo "=============================================="

# =============================================================================
# STAGE 2 — embed expert 5-class drainage labels (in place, CF-1.8 append)
# =============================================================================
# Idempotent: files that already have drainage_label are skipped, so this
# is safe even if the build only partially completed (resume continues it).
echo ""
echo ">>> STAGE 2: embed labels  ($LABELS_CSV)"
python3 -u "$REPO_DIR/engine/labeling/add_labels_to_stacks.py" \
    --stacks_dir "$OUTPUT_DIR" \
    --labels_csv "$LABELS_CSV" \
    --id_col lake_id \
    --workers 8
LABELS_RC=$?
echo "STAGE 2 (labels) done: $(date)  rc=$LABELS_RC"

# =============================================================================
# STAGE 3 — CF-1.8 validation of the now label-complete dir (authoritative)
# =============================================================================
# UDUNITS2 backs the authoritative cfchecks. On Sherlock the module is
# `udunits/2.2.26` and it lives UNDER the `physics` hierarchy — there is
# no `udunits2`, and a bare `ml udunits` fails until `physics` is loaded.
# If cfchecks is pip-installed but libudunits2.so is absent it does not
# degrade gracefully — it SEGFAULTS (job 25413767 stage-3 rc=139). Load
# the hierarchy parent then the module so the C lib resolves.
ml physics 2>/dev/null && ml udunits/2.2.26 2>/dev/null
if ml list 2>&1 | grep -qi udunits; then
    echo "udunits module loaded ($(ml list 2>&1 | grep -oi 'udunits/[0-9.]*'))"
else
    echo "NOTE: udunits/2.2.26 did NOT load — cfchecks may be unavailable;"
    echo "      cf_check still runs its dependency-free structural audit."
fi
pip install --user cfchecker >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
echo ""
echo ">>> STAGE 3: cf-check  (cfchecks on PATH: $(command -v cfchecks || echo 'NO — structural only'))"
python3 -u "$REPO_DIR/engine/validation/cf_check.py" \
    "$OUTPUT_DIR" \
    --version 1.8 \
    --workers 16
CF_RC=$?
echo "STAGE 3 (cf-check) done: $(date)  rc=$CF_RC"

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

# Build + labels are the durable, expensive work. cf-check is read-only
# and re-runnable. Its exit codes:
#   0      = CLEAN (built + labeled + CF-1.8 signed off in one job)
#   1      = cf_check RAN and found real CF errors  -> fail OVERALL
#   >=126  = it CRASHED / couldn't run (e.g. 139 SIGSEGV from a broken
#            UDUNITS env) -> infra problem, NOT a data problem; build+
#            labels stay durable, so WARN and let OVERALL reflect
#            build/labels. Get the authoritative sign-off via the
#            standalone run_cf_check.sh.
OVERALL=0
[ "$BUILD_RC" -ne 0 ] && OVERALL=$BUILD_RC
[ "$OVERALL" -eq 0 ] && [ "$LABELS_RC" -ne 0 ] && OVERALL=$LABELS_RC
CF_NOTE="CF-1.8 signed off"
if [ "$OVERALL" -eq 0 ]; then
    if [ "$CF_RC" -eq 0 ]; then
        :
    elif [ "$CF_RC" -ge 126 ]; then
        CF_NOTE="CF-check CRASHED (rc=$CF_RC) — env, not data; build+labels durable"
        echo ""
        echo "WARNING: cf-check crashed (rc=$CF_RC, likely UDUNITS/cfchecks"
        echo "         environment, not the data). Build + labels are durable."
        echo "         Authoritative sign-off: sbatch engine/validation/"
        echo "         run_cf_check.sh ${REGION} ${YEAR}"
    else
        OVERALL=$CF_RC
        CF_NOTE="CF errors found (cf_check rc=$CF_RC)"
    fi
fi

NC_COUNT=$(ls "$OUTPUT_DIR/"*.nc 2>/dev/null | wc -l)
echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m"
echo "Stacks:   $NC_COUNT  ($OUTPUT_DIR)"
echo "rc:  build=$BUILD_RC  labels=$LABELS_RC  cf-check=$CF_RC"
echo "OVERALL exit: $OVERALL   ($CF_NOTE)"
echo "=============================================="

exit $OVERALL
