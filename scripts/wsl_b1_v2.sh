#!/bin/bash
# Build the B1 v2 circuit and verify it with the official verifier (WSL).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation

echo "=== build v2 circuit ==="
$PY src/b1_v2_build.py results/b1_v2.kmb --wc 17 --db 12 --wsq 28 --wz 28 --wp 28

echo "=== official verifier (full 200M shots) ==="
cd data/verifiers
$PY invsqrt/verify_submission_invsqrt.py /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1_v2.kmb 2>&1 | tee /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1_v2_score.json | tail -20
