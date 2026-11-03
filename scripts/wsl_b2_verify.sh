#!/bin/bash
# Build the B2 (DQI) decoder circuit and verify it with the official verifier.
#
# Usage:
#   bash scripts/wsl_b2_verify.sh                # full scale, 102,400 shots
#   bash scripts/wsl_b2_verify.sh --num-shots 4096 --no-build
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
ROOT=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
cd "$ROOT"

NUM_SHOTS=102400
BUILD=1
for arg in "$@"; do
  case "$arg" in
    --num-shots) NEXT_SHOTS=1 ;;
    --no-build) BUILD=0 ;;
    *) if [ "${NEXT_SHOTS:-0}" = "1" ]; then NUM_SHOTS="$arg"; NEXT_SHOTS=0; fi ;;
  esac
done

if [ "$BUILD" = "1" ]; then
  echo "=== building dqi.kmb (full scale, may take several minutes) ==="
  $PY src/b2_build.py results/b2_dqi.kmb
fi

echo "=== official dqi verifier (num_shots=$NUM_SHOTS) ==="
cd data/verifiers
$PY dqi/verify_submission_dqi.py "$ROOT/results/b2_dqi.kmb" --num-shots "$NUM_SHOTS" \
  2>&1 | tee "$ROOT/results/b2_dqi_score.json" | tail -25
