#!/bin/bash
# Long (6,192,68) route, segmented: each 3000-epoch segment seeds from the previous.
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
PY=/home/kaggle/miniconda3/envs/py313/bin/python
SEED=""
OUT=results/b3_train_6_192_68_long.npz
for i in 1 2 3 4 5; do
  if [ -f "$OUT" ]; then SEED="--seed-solution $OUT"; fi
  $PY -u src/b3_train2.py --rank 6 --bases 192 --copies 68 --epochs 3000 --lr 0.001 \
    --rho 1e-4 --epsilon 0.010 --seed $i $SEED --out $OUT \
    2>&1 | tee results/b3_train_6_192_68_long.log
done
