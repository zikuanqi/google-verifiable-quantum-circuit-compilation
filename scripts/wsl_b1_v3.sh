#!/bin/bash
# B1 v3: fit (if needed), build, quick test, then official 200M-shot verification (WSL).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
R=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
cd "$R"

TABS=${1:-$R/results/b1v3_tabs_8_19_27_1_d5.npz}
KMB=${2:-$R/results/b1_v3.kmb}
JSON=${3:-$R/results/b1_v3_score.json}

echo "=== build v3 circuit ==="
$PY scripts/b1v3_build_test.py "$TABS" "$KMB"

echo "=== quick functional test ==="
$PY src/b1v3_test.py "$TABS" "$KMB"

echo "=== official verifier (full 200M shots) ==="
cd data/verifiers
$PY invsqrt/verify_submission_invsqrt.py "$KMB" 2>&1 | tee "$JSON" | tail -20

echo "=== ALL DONE ==="
