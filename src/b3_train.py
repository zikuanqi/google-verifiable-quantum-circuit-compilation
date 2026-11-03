"""Driver for the official DFTHC (Qaggle) trainer on FeMoCo76o113e.

Imports the starter kit's trainer and runs a configurable (rank, bases,
copies) job, seeding from the packaged baseline. Output goes to results/.

Usage (in WSL, py313, PYTHONPATH set to the starter kit):
    python b3_train.py --rank 6 --bases 192 --copies 68 --epochs 200 --lr 0.003 \
        --out results/b3_train_6_192_68.npz
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
        print_interval=max(1, args.epochs // 10),
        rank=args.rank,
        bases=args.bases,
        copies=args.copies,
        epsilon_dfthc=args.epsilon,
        output_file=args.out,
    )
    print(f"done in {time.time() - t0:.0f}s")
    print({k: (float(v) if hasattr(v, "__float__") else v) for k, v in values.items()
           if k in ("error_h2", "lambda_eff", "Lambda", "lambda_H1", "lambda_H2",
                    "E_SOS", "sos_gap", "cost")})


if __name__ == "__main__":
    main()
