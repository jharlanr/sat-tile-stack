#!/bin/bash
#SBATCH --job-name=sts_cfcheck
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH -p serc,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# CF-1.8 VALIDATION FOR THE v2 STACKS  (run AFTER the builds finish)
# =============================================================================
#
# Runs sat_tile_stack.cf_check (dependency-free structural audit + the
# standard `cfchecks` when available) over the per-lake NetCDFs in
# stacks_v2/<REGION>_<YEAR>/, plus a count sanity check.
#
# USAGE (defaults to CW 2018 + CW 2019):
#   sbatch engine/validation/run_cf_check.sh
#   sbatch engine/validation/run_cf_check.sh CW 2018          # one year
#   sbatch engine/validation/run_cf_check.sh CW 2018 CW 2019  # explicit pairs
#
# Submit this once CW 2019 has finished. If the 2019 BUILD timed out,
# re-`sbatch run_build_stacks_region.sh CW 2019` first (resume-safe:
# skips complete lakes, rebuilds partials), then run this.
# =============================================================================

set -uo pipefail

REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/sat-tile-stack"
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack"
STACKS_DIR="$SHERLOCK_DIR/stacks_v2"

# (REGION YEAR) pairs to check — default CW 2018 + CW 2019.
if [ "$#" -ge 2 ]; then
    PAIRS=("$@")
else
    PAIRS=(CW 2018 CW 2019)
fi

# Expected lake counts (centroid CSVs are authoritative; these are the
# documented targets for the sanity print).
declare -A EXPECTED=( ["CW_2018"]=679 ["CW_2019"]=1000 )

mkdir -p "$SHERLOCK_DIR/logs"

# --- Modules / deps (mirror run_build_stacks_region.sh) ---
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
# UDUNITS2 backs the standard cfchecker; load it if a module exists so the
# authoritative checker runs (otherwise cf_check gracefully degrades to the
# dependency-free structural audit only).
ml udunits 2>/dev/null || ml udunits2 2>/dev/null || \
    echo "NOTE: no udunits module found — cfchecker may be unavailable; " \
         "structural audit still runs."

pip install --user xarray netcdf4 pyproj cfchecker >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

echo "=============================================="
echo "sat-tile-stack: CF-1.8 validation"
echo "Start: $(date)"
echo "cfchecks on PATH: $(command -v cfchecks || echo 'NO (structural only)')"
echo "=============================================="

# --- Collect target dirs + count sanity check ---
TARGETS=()
i=0
while [ $i -lt ${#PAIRS[@]} ]; do
    REGION="${PAIRS[$i]}"; YEAR="${PAIRS[$((i+1))]}"; i=$((i+2))
    KEY="${REGION}_${YEAR}"
    DIR="$STACKS_DIR/$KEY"
    if [ ! -d "$DIR" ]; then
        echo "ERROR: $DIR not found — skipping $KEY"
        continue
    fi
    N=$(find "$DIR" -name '*.nc' | wc -l | tr -d ' ')
    EXP="${EXPECTED[$KEY]:-?}"
    echo "  $KEY: $N .nc files (expected ~$EXP)"
    if [ "$EXP" != "?" ] && [ "$N" -lt "$EXP" ]; then
        echo "  WARNING: $KEY is INCOMPLETE ($N < $EXP) — the build may not"
        echo "           have finished; validating what is present anyway."
    fi
    TARGETS+=("$DIR")
done

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "ERROR: no valid target directories. Aborting."
    exit 1
fi

echo ""
echo "Running cf_check on: ${TARGETS[*]}"
echo ""

python3 -u "$REPO_DIR/engine/validation/cf_check.py" \
    "${TARGETS[@]}" \
    --version 1.8 \
    --workers 16
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End: $(date)"
echo "cf_check exit code: $EXIT_CODE  (0 = ALL CLEAN, 1 = failures)"
echo "=============================================="
exit $EXIT_CODE
