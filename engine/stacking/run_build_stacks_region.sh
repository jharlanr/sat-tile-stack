#!/bin/bash
#SBATCH --job-name=sts_build
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# BUILD TIMESTACKS FOR ANY REGION + YEAR
# =============================================================================
#
# Builds 512x512 @ 10m daily timestacks (May-Sep) from Dunmire GeoJSON.
# Filters by IMBIE region, computes polygon centroids, then runs
# build_stacks.py with consistent parameters.
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
#
# =============================================================================

set -euo pipefail

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
OUTPUT_DIR="$SHERLOCK_DIR/stacks/${REGION}_${YEAR}"
EXTRACT_CSV="$SHERLOCK_DIR/stacks/${REGION}_${YEAR}_centroids.csv"

mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "sat-tile-stack: Build ${REGION} ${YEAR} Stacks"
echo "=============================================="
echo "Region:     $REGION"
echo "Year:       $YEAR"
echo "GeoJSON:    $DUNMIRE_GEOJSON"
echo "Output:     $OUTPUT_DIR"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
echo "=============================================="

# --- Load modules ---
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

pip install --user xarray netcdf4 pystac-client planetary-computer stackstac geopandas rioxarray pyproj shapely matplotlib dask

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# --- Extract region CSV with centroids from Dunmire GeoJSON ---
echo ""
echo "Extracting ${REGION} lakes from Dunmire GeoJSON..."

python3 -u -c "
import json
import csv
import sys

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

    # Compute centroid from polygon coordinates
    coords = feat['geometry']['coordinates']
    # Handle Polygon (single ring) — use outer ring
    ring = coords[0]
    n = len(ring)
    lon_c = sum(c[0] for c in ring) / n
    lat_c = sum(c[1] for c in ring) / n

    row = dict(props)
    row['lon'] = lon_c
    row['lat'] = lat_c
    # Store geometry as WKT-ish string for reference (not used by build_stacks)
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

print(f'  Extracted {len(rows)} lakes for {region} {geojson_path.split(\"_\")[1][:4]}')
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
    --bands B04 B03 B02 B08 B11 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 1 \
    --count 1

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
    --bands B04 B03 B02 B08 B11 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 8

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m"
echo "Exit code: $EXIT_CODE"

NC_COUNT=$(ls "$OUTPUT_DIR/"*.nc 2>/dev/null | wc -l)
echo "Stacks built: $NC_COUNT"
echo "=============================================="

exit $EXIT_CODE
