#!/bin/bash
# Verify the public 19.5-score submission with the official verifiers (local, WSL).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
V=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers
P=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_19_53750_public_resources/run012a_public_release/payload
cd "$V"

echo "=== invsqrt (their 403KB circuit) ==="
$PY invsqrt/verify_submission_invsqrt.py "$P/invsqrt.kmb" 2>&1 | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ('status','score','toffoli_equivalent','toffoli_count','num_qubits') if k in d}, indent=2))"

echo "=== dqi (their 282MB circuit) ==="
$PY dqi/verify_submission_dqi.py "$P/dqi.kmb" 2>&1 | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ('status','score','toffoli_count','num_qubits','empirical_error_rate','effective_error_rate','repetition_factor','value_errors','phase_errors') if k in d}, indent=2))"

echo "=== dfthcsos ==="
$PY dfthcsos/verify.py "$P/dfthcsos" 2>&1 | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ('status','score','lambda_eff','toffoli_count','num_qubits') if k in d}, indent=2))"
