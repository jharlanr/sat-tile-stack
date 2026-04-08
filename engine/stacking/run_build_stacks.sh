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
# BUILD TIMESTACKS FOR CW 2018
# =============================================================================
#
# Builds 512x512 @ 10m daily timestacks (May-Sep) for all CW 2018 lakes.
# Uses 32 parallel workers hitting Planetary Computer.
#
# USAGE:
#   sbatch run_build_stacks.sh
#
# =============================================================================

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/sat-tile-stack"
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack"

mkdir -p "$SHERLOCK_DIR/logs"

echo "=============================================="
echo "sat-tile-stack: Build CW 2018 Stacks"
echo "=============================================="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=============================================="

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312

# Install dependencies
pip install --user xarray netcdf4 pystac-client planetary-computer stackstac geopandas rioxarray pyproj shapely matplotlib dask

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo ""
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# --- Build first stack solo and inspect it ---
echo "Building first stack (single worker) for inspection..."
python3 -u "$REPO_DIR/engine/stacking/build_stacks.py" \
    --csv "$REPO_DIR/labeling/labels/labels_2018_volumes_CW.csv" \
    --output_dir "$SHERLOCK_DIR/stacks/CW_2018" \
    --id_col "new_id" \
    --time_range "2018-05-01/2018-09-30" \
    --bands B04 B03 B02 B08 B11 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 1 \
    --count 1

# Inspect the first file
FIRST_NC=$(ls "$SHERLOCK_DIR/stacks/CW_2018/"*.nc 2>/dev/null | head -1)
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
echo "Building remaining stacks (32 workers)..."
python3 -u "$REPO_DIR/engine/stacking/build_stacks.py" \
    --csv "$REPO_DIR/labeling/labels/labels_2018_volumes_CW.csv" \
    --output_dir "$SHERLOCK_DIR/stacks/CW_2018" \
    --id_col "new_id" \
    --time_range "2018-05-01/2018-09-30" \
    --bands B04 B03 B02 B08 B11 SCL \
    --pix_res 10 \
    --tile_size 512 \
    --cloudmask scl \
    --workers 32

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

NC_COUNT=$(ls "$SHERLOCK_DIR/stacks/CW_2018/"*.nc 2>/dev/null | wc -l)
echo "Stacks built: $NC_COUNT"
echo "=============================================="

exit $EXIT_CODE
