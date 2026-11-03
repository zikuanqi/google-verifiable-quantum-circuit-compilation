#!/bin/bash
# Probe the derivative-split builds at several split values (WSL).
set -e
R=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_6_17493_sources
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:$R/src/invsqrt
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd "$R"
for sp in 8 9 10 11 12 13; do
  echo "=== split $sp ==="
  $PY src/invsqrt/run002_signed_build.py /tmp/b1v3_ds_$sp.kmb \
    --tables runs/invsqrt/run002/virtual_derivative_refined_coefficients.json \
    --derivative-split $sp --virtual-low-bit --virtual-derivative-bit 2>&1 | \
    $PY -c 'import json,sys; d=json.load(sys.stdin); print("qubits",d["num_qubits"],"static_toffoli",d["max_toffoli_count"],"rot",d["rotation_count"],"ops",d["num_ops"],"split",d["derivative_split"])'
done
