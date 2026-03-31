#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/prove_flatone_pipeline.sh [SEGMENT_ID] [OUTPUT_DIR]
#
# Example:
#   bash scripts/prove_flatone_pipeline.sh 720575940581355117 outputs/proof

SEG_ID="${1:-720575940581355117}"
OUTPUT_DIR="${2:-outputs/proof}"
PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found or not executable."
  echo "Create the venv first and install dependencies."
  exit 2
fi

RUN_DIR="${OUTPUT_DIR}/run_${SEG_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

TOKEN_LOG="$RUN_DIR/token_check.log"
PIPELINE_LOG="$RUN_DIR/flatone_pipeline.log"
VERIFY_LOG="$RUN_DIR/verify_outputs.log"

echo "== [1/4] Token/API check =="
"$PY" src/check_eyewire_token.py --datastack stroeh_mouse_retina | tee "$TOKEN_LOG"

echo
echo "== [2/4] Run Flatone pipeline =="
"$PY" flatone/flatone/cli.py "$SEG_ID" --output-dir "$OUTPUT_DIR" --warp-mesh | tee "$PIPELINE_LOG"

echo
echo "== [3/4] Mapping file used (default j2) =="
MAPPING_PATH=$(PYTHONPATH=flatone "$PY" - <<'PY'
from flatone.cli import _resolve_mapping_file
print(_resolve_mapping_file("j2"))
PY
)
echo "$MAPPING_PATH" | tee -a "$PIPELINE_LOG"

echo
echo "== [4/4] Verify generated files =="
SEG_DIR="${OUTPUT_DIR}/${SEG_ID}"
required=(
  "mesh.obj"
  "mesh_warped.obj"
  "skeleton.swc"
  "skeleton.npz"
  "skeleton.png"
  "skeleton_warped.swc"
  "skeleton_warped.npz"
  "skeleton_warped.png"
  "strat_profile.png"
)

for f in "${required[@]}"; do
  p="${SEG_DIR}/${f}"
  if [[ ! -s "$p" ]]; then
    echo "MISSING_OR_EMPTY: $p" | tee -a "$VERIFY_LOG"
    exit 1
  fi
  size=$(stat -f%z "$p")
  echo "OK: $p (${size} bytes)" | tee -a "$VERIFY_LOG"
done

echo
echo "All checks passed."
echo "Token log:    $TOKEN_LOG"
echo "Pipeline log: $PIPELINE_LOG"
echo "Verify log:   $VERIFY_LOG"
echo "Output files:"
ls -lah "$SEG_DIR"
