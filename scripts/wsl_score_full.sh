#!/bin/bash
# Full official scoring runs (WSL).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
V=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers
S=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission
OUT=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results
mkdir -p "$OUT"
cd "$V"

echo "=== Benchmark III full summary ==="
$PY dfthcsos/verify.py "$S/dfthcsos" 2>&1 | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in ('status','score','lambda_eff','toffoli_count','num_qubits','verified_shots','submission_sha256') if k in d}, indent=2))"

echo "=== Benchmark I full (64 shots/x, 200M shots) ==="
$PY invsqrt/verify_submission_invsqrt.py "$S/invsqrt.kmb" 2>&1 | tee "$OUT/b1_sample_score.json" | tail -20

echo "=== Benchmark II full (102,400 shots) ==="
$PY dqi/verify_submission_dqi.py "$S/dqi.kmb" 2>&1 | tee "$OUT/b2_sample_score.json" | tail -20

echo "=== ALL DONE ==="
