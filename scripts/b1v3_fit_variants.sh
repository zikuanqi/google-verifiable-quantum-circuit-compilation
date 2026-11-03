#!/bin/bash
# Fit several v3 config variants sequentially (WSL).
set -e
PY=/home/kaggle/miniconda3/envs/py313/bin/python
F=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/scripts/b1v3_fit.py
for tag in 8_19_27_1_d5,6 8_18_27_1_d5 7_19_27_1_d5 8_19_27_1_d5,6,7; do
  echo "=== $tag ==="
  $PY $F $tag
done
echo ALL_FITS_DONE
