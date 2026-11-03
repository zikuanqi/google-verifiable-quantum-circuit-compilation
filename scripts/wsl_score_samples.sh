#!/bin/bash
# Score the official sample submissions with the official verifiers (WSL).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
V=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers
S=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission
cd "$V"

echo "=== Benchmark I (invsqrt) quick sanity: 1 shot per x ==="
$PY invsqrt/verify_submission_invsqrt.py "$S/invsqrt.kmb" --shots-per-x 1 2>&1 | tail -12

echo ""
echo "=== Benchmark III (dfthcsos) sample ==="
$PY dfthcsos/verify.py "$S/dfthcsos" 2>&1 | tail -12
