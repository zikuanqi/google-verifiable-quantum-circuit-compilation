# -*- coding: utf-8 -*-
"""Benchmark I v2 sweep: K=32 interval / D=2 polynomial + NR, multiprocess.

Evaluates the full integer model over ALL x for each width config, reports
worst-case relative error, phase margin, and analytic Toffoli/Z_POW cost.
Run on Windows with plain numpy (b1_v2_model + b1_fit only).
"""
from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_v2_model import V2Precision, v2_model  # noqa: E402

N_P = 10
WIDTH = 2 * N_P + 2
MAX_X = 3 * (2**N_P - 2)**2
TARGET_BITS = 24
FEAS_MARGIN = 0.4
K = 32
D = 2


def toffoli_estimate(wc, db, wsq, wz, wp):
    """Analytic per-direction Toffoli estimate (v2: 2 Horner mults, db rows)."""
    width = WIDTH
    t = 0.0
    t += 10 + 3 + 30          # normalize flags + select interval + coefficient ANDs
    m0 = (K - 1).bit_length() - 1
    t += D * sum(2 * (wc - m) for m in range(m0, min(db, width - 2) + 1))  # horner
    for k in range(1, wc + 1):   # square
        nterms = max(0, min(wc, wsq - k + 1) - k)
        if nterms:
            t += nterms + 2 * k
    for i in range(width):       # multiply_m
        e = wz - wsq + i - width + 2
        nterms = wsq - max(0, -e)
        if nterms > 0:
            t += nterms + (wz + 2 - max(0, e))
    t += wz + 2                  # correct
    for k in range(1, wc + 1):   # multiply_y0
        e = wp - wz - k
        nterms = wz + 2 - max(0, -e)
        if nterms > 0:
            t += nterms + (wp + 1 - max(0, -e))
    return t


def qubit_count(wc, wsq, wz, wp):
    out_bits = wp + N_P + 1
    return WIDTH + out_bits + N_P + 3 + (D + 1) * wc + wsq + (wz + 2) \
        + max(wsq, wz + 2, wp + 1)


def zpow_t_cost(n_rot, min_margin_turns):
    eps = math.pi * min_margin_turns / n_rot
    return 1.0 + n_rot * (0.56 * math.log2(1.0 / eps) + 5.3)


def eval_config(cfg):
    wc, db, wsq, wz, wp = cfg
    prec = V2Precision(K=K, degree=D, coef_bits=wc, delta_bits=db,
                       square_bits=wsq, product_bits=wz, result_bits=wp)
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    inv_sqrt = 1.0 / np.sqrt(xs.astype(np.float64))
    try:
        y1 = v2_model(N_P, prec, xs)
        rel = np.abs(y1 / 2.0**prec.output_bits(N_P) - inv_sqrt) / inv_sqrt
        worst = float(rel.max())
        margin = (2.0**-TARGET_BITS - rel) * inv_sqrt
        min_margin = float(margin.min())
        ok = worst <= 2 ** -(TARGET_BITS + FEAS_MARGIN) and min_margin > 0
    except OverflowError:
        worst, ok, min_margin = float("inf"), False, 0.0
    n_rot = wp + 8
    zt = zpow_t_cost(n_rot, min_margin) if ok else float("inf")
    cost_t = toffoli_estimate(wc, db, wsq, wz, wp) * 2
    q = qubit_count(wc, wsq, wz, wp)
    equiv = cost_t + zt / 4
    return (worst, ok, equiv, cost_t, zt, q, cfg)


def main():
    configs = []
    for wc, db, wsq, wz, wp in itertools.product(
            (15, 16, 17, 18), (10, 12, 14), (24, 26, 28), (24, 26, 28),
            (25, 26, 27, 28, 29)):
        if not (wsq >= wc and wz >= wsq - 4 and wp >= wz - 2):
            continue
        configs.append((wc, db, wsq, wz, wp))
    print(f"{len(configs)} configs, {mp.cpu_count()} cpus", flush=True)

    results = []
    out_npz = Path(__file__).resolve().parent.parent / "results" / "b1_sweep_v2.npz"
    with mp.Pool(processes=min(10, mp.cpu_count())) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_config, configs)):
            worst, ok, equiv, cost_t, zt, q, cfg = r
            results.append(r)
            if ok:
                print(f"[{i+1}/{len(configs)}] {cfg}: worst=2^{math.log2(worst):.2f} "
                      f"OK eqv={equiv:.0f} estT={cost_t:.0f} ZpowT={zt:.0f} Q={q}",
                      flush=True)
            else:
                print(f"[{i+1}/{len(configs)}] {cfg}: worst=2^{math.log2(worst):.2f} FAIL",
                      flush=True)
            if (i + 1) % 25 == 0:  # checkpoint for crash safety
                np.savez(out_npz,
                         results=np.array(
                             [(r[0], r[1], r[2], r[3], r[4], r[5], *r[6]) for r in results],
                             dtype=np.float64))

    feas = [r for r in results if r[1]]
    feas.sort(key=lambda r: (r[2] * r[5], r[2]))
    print("\n=== feasible by eqv*Q ===")
    for worst, ok, equiv, cost_t, zt, q, cfg in feas[:20]:
        print(f"  eqv*Q={equiv*q:>9.0f}  eqv={equiv:>7.1f}  estT={cost_t:>6.0f} "
              f"ZpowT={zt:>5.1f}  Q={q:>3d}  worst=2^{math.log2(worst):.2f}  cfg={cfg}")
    np.savez(out_npz,
             results=np.array([(r[0], r[1], r[2], r[3], r[4], r[5], *r[6]) for r in results],
                              dtype=np.float64))
    print("saved results/b1_sweep_v2.npz")


if __name__ == "__main__":
    main()
