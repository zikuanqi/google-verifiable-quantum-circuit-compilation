#!/bin/bash
# Route A: (65,60,30) DFTHC factorization, warm-started from the packaged baseline.
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
/home/kaggle/miniconda3/envs/py313/bin/python src/b3_train.py \
  --rank 65 --bases 60 --copies 30 --epochs 2000 --lr 0.001 \
  --rho 1e-4 --epsilon 0.010 --out results/b3_train_65_60_30.npz \
  2>&1 | tee results/b3_train_65_60_30.log
