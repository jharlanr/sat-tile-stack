#!/bin/bash
#SBATCH --job-name=sts_add_labels
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# EMBED EXPERT 5-CLASS DRAINAGE LABELS INTO THE v2 STACKS  (post-build pass)
# =============================================================================
#
# The build jobs did NOT add labels (coregister chain = p_water /
# lake_boundary / water_mask_ndwi only). This adds, in place and CF-1.8:
#   * drainage_label      scalar int8 CF flag variable (hard label)
#   * label_probability   (class) float32 soft vector + class/class_name
#   * notes / flagged     CF-sanitised global attrs
# 3-rater inter-rater labels stay in the sidecar IRR CSVs (by design).
#
# Resume-safe / idempotent: files that already have drainage_label skip.
# No network. Run AFTER both year builds finish, BEFORE run_cf_check.sh
# (so cf_check validates the label-complete files).
#
# USAGE (defaults to CW 2018 + CW 2019):
#   sbatch engine/labeling/run_add_labels.sh
#   sbatch engine/labeling/run_add_labels.sh CW 2018
#
# Override the labels-CSV location with LABELS_DIR=... (must contain
# labels_<REGION>_<YEAR>.csv).
# =============================================================================

set -uo pipefail

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/sat-tile-stack"
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack"
STACKS_DIR="$SHERLOCK_DIR/stacks_v2"
# Canonical 5-class label CSVs (lake_id,label,p_ND..p_CD,notes,flagged).
# Override at submit: LABELS_DIR=/path sbatch engine/labeling/run_add_labels.sh
LABELS_DIR="${LABELS_DIR:-/oak/stanford/groups/cyaolai/JoshRines/data/labels}"

if [ "$#" -ge 2 ]; then
    PAIRS=("$@")
else
    PAIRS=(CW 2018 CW 2019)
fi
declare -A EXPECTED=( ["CW_2018"]=679 ["CW_2019"]=1000 )

mkdir -p "$SHERLOCK_DIR/logs"

ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
pip install --user xarray netcdf4 pyproj >/dev/null 2>&1 || true
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "=============================================="
echo "sat-tile-stack: embed drainage labels"
echo "Start: $(date)   LABELS_DIR=$LABELS_DIR"
echo "=============================================="

# --- Pre-flight: every target dir + its labels CSV must exist ---
TASKS=()
i=0
while [ $i -lt ${#PAIRS[@]} ]; do
    REGION="${PAIRS[$i]}"; YEAR="${PAIRS[$((i+1))]}"; i=$((i+2))
    KEY="${REGION}_${YEAR}"
    DIR="$STACKS_DIR/$KEY"
    CSV="$LABELS_DIR/labels_${KEY}.csv"
    if [ ! -d "$DIR" ]; then
        echo "ERROR: stacks dir missing: $DIR"; exit 1
    fi
    if [ ! -f "$CSV" ]; then
        echo "ERROR: labels CSV missing: $CSV"
        echo "Stage the canonical labels_${KEY}.csv there or pass LABELS_DIR=."
        exit 1
    fi
    N=$(find "$DIR" -name '*.nc' | wc -l | tr -d ' ')
    EXP="${EXPECTED[$KEY]:-?}"
    echo "  $KEY: $N .nc (expected ~$EXP)  <- $CSV"
    if [ "$EXP" != "?" ] && [ "$N" -lt "$EXP" ]; then
        echo "  WARNING: $KEY incomplete ($N < $EXP) — labeling what is"
        echo "           present; rerun this job after the build finishes"
        echo "           (idempotent: already-labeled files are skipped)."
    fi
    TASKS+=("$DIR|$CSV")
done

EXIT_CODE=0
for t in "${TASKS[@]}"; do
    DIR="${t%%|*}"; CSV="${t##*|}"
    echo ""
    echo ">>> $DIR"
    python3 -u "$REPO_DIR/engine/labeling/add_labels_to_stacks.py" \
        --stacks_dir "$DIR" \
        --labels_csv "$CSV" \
        --id_col lake_id \
        --workers 8
    rc=$?
    [ $rc -ne 0 ] && EXIT_CODE=$rc
done

echo ""
echo "=============================================="
echo "End: $(date)   exit=$EXIT_CODE"
echo "Next: sbatch engine/validation/run_cf_check.sh  (validate label-complete files)"
echo "=============================================="
exit $EXIT_CODE
