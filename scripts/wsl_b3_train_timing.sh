#!/bin/bash
# Quick timing test of the official DFTHC trainer on (6,192,68), few epochs.
set -e
PY=/home/kaggle/miniconda3/envs/py313/bin/python
SK=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/starter_kit/starter_kit
cd "$SK"
export PYTHONPATH="$SK:$SK/.."
OUT=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b3_train_timing_6_192_68.npz
rm -f "$OUT"
$PY starter/task1_factorization/train_df_thc_qaggle.py \
    data/integrals/FeMoCo76o113e.fcidump \
    --system FeMoCo76o113e \
    --rank 6 --bases 192 --copies 68 \
    --epochs 60 --learning-rate 0.003 \
    --fileout "$OUT" 2>&1 | tail -20
