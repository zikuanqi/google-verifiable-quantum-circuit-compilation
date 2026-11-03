# -*- coding: utf-8 -*-
"""DFTHC trainer driver with explicit seed control (b3-opt's variant of b3_train.py).

Usage (in WSL, py313):
    python src/b3_train2.py --rank 65 --bases 60 --copies 30 --epochs 5000 --lr 0.001 \
        --rho 0.001 --epsilon 0.010 --seed-solution results/b3_train_65_60_30.npz \
        --out results/b3_train_65_60_30_v2.npz
"""
from __future__ import annotations

import argparse
import sys
import time

STARTER = "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/starter_kit/starter_kit"
if STARTER not in sys.path:
    sys.path.insert(0, STARTER)
    sys.path.insert(0, STARTER + "/..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--bases", type=int, required=True)
    ap.add_argument("--copies", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--rho", type=float, default=1e-4)
    ap.add_argument("--epsilon", type=float, default=0.010)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-solution", default=None, help="npz to warm-start from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from starter.task1_factorization.train_df_thc_qaggle import train_system

    t0 = time.time()
    params, values = train_system(
        "FeMoCo76o113e",
        epochs=args.epochs,
        lr=args.lr,
        rho=args.rho,
        seed=args.seed,
        print_interval=max(1, args.epochs // 20),
        rank=args.rank,
        bases=args.bases,
        copies=args.copies,
        epsilon_dfthc=args.epsilon,
        seed_solution_file=args.seed_solution,
        output_file=args.out,
    )
    print(f"done in {time.time() - t0:.0f}s")
    for k in ("error_h2", "epsilon_fro", "lambda_eff", "Lambda", "lambda_H1",
              "lambda_H2", "E_SOS", "sos_gap", "cost"):
        if k in values:
            v = values[k]
            print(f"{k}: {float(v):.6g}" if hasattr(v, "__float__") else f"{k}: {v}")


if __name__ == "__main__":
    main()
