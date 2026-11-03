#!/bin/bash
# Continuation run segment 1: warm-start from the previous trained state, stronger lambda_eff pressure.
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
/home/kaggle/miniconda3/envs/py313/bin/python -u src/b3_train2.py \
  --rank 65 --bases 60 --copies 30 --epochs 1500 --lr 0.001 \
  --rho 0.001 --epsilon 0.010 --seed 1 \
  --seed-solution results/b3_train_65_60_30.npz \
  --out results/b3_train_65_60_30_v2.npz \
  2>&1 | tee results/b3_train_65_60_30_v2.log
