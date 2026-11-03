"""Benchmark I: sweep Precision bit widths with the reference fixed-point model.

Runs the official fixed_point_model over ALL x in [1, max_x] for candidate
Precision configs, reports worst-case relative error, phase margin, and an
analytic cost estimate (Toffolis + Z_POW synthesis T-cost). Feasible configs
must keep worst rel error below 2^-24 with margin.
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_model import Precision, fixed_point_model  # noqa: E402

N_P = 10
WIDTH = 2 * N_P + 2          # 22
MAX_X = 3 * (2**N_P - 2)**2  # 3,133,452
TARGET_BITS = 24             # relative precision
FEAS_MARGIN = 0.4            # require worst error <= 2^-(24+margin)


def toffoli_estimate(prec: Precision) -> float:
    """Analytic per-direction Toffoli estimate of the reference circuit structure."""
    d, wc, wsq, wz, wp = prec.delta_bits, prec.coef_bits, prec.square_bits, \
        prec.product_bits, prec.result_bits
    width = WIDTH
    db = min(d, width - 2)
    t = 0.0
    t += 10 + 3  # normalize flags + select interval
    # horner: 3 steps, per delta bit m: 2*(wc - m) (AND + add)
    t += 3 * sum(2 * (wc - m) for m in range(1, db + 1))
    # square: cross terms k<l with k+l-1 <= wsq
    for k in range(1, wc + 1):
        nterms = max(0, min(wc, wsq - k + 1) - k)
        if nterms:
            t += nterms + 2 * k
    # multiply_m: per bit i of x_norm
    for i in range(width):
        e = wz - wsq + i - width + 2
        nterms = wsq - max(0, -e)
        if nterms > 0:
            t += nterms + (wz + 2 - max(0, e))
    # correct: 3 - z
    t += wz + 2
    # multiply_y0: per bit k of y0
    for k in range(1, wc + 1):
        e = wp - wz - k
        nterms = wz + 2 - max(0, -e)
        if nterms > 0:
            t += nterms + (wp + 1 - max(0, -e))
    return t


def qubit_count(prec: Precision) -> int:
    d, wc, wsq, wz, wp = prec.delta_bits, prec.coef_bits, prec.square_bits, \
        prec.product_bits, prec.result_bits
    out_bits = prec.output_bits(N_P)
    return WIDTH + out_bits + N_P + 3 + 4 * wc + wsq + (wz + 2) \
        + max(wsq, wz + 2, wp + 1)


def zpow_t_cost(n_rot: int, min_margin_turns: float) -> float:
    """T-count of the Z_POW layer: 1 T gate + n_rot general rotations.

    eps = pi * min_margin / n_rot; each general rotation costs
    0.56 log2(1/eps) + 5.3 T (mixed fallback synthesis).
    """
    eps = math.pi * min_margin_turns / n_rot
    return 1.0 + n_rot * (0.56 * math.log2(1.0 / eps) + 5.3)


def main():
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    inv_sqrt = 1.0 / np.sqrt(xs.astype(np.float64))
    print(f"x in [1, {MAX_X}], target <= 2^-{TARGET_BITS + FEAS_MARGIN:.1f}")

    configs = []
    for d, wc, wsq, wz, wp in itertools.product(
        (8, 10, 12, 14),
        (14, 15, 16, 17, 18),
        (20, 22, 24, 26, 28),
        (22, 24, 26, 28, 30),
        (24, 25, 26, 27, 28, 29, 30, 31),
    ):
        if not (wsq >= wc and wz >= wsq - 4 and wp >= wz - 2):
            continue
        configs.append((d, wc, wsq, wz, wp))
    print(f"{len(configs)} configs to evaluate")

    results = []
    for idx, (d, wc, wsq, wz, wp) in enumerate(configs):
        prec = Precision(delta_bits=d, coef_bits=wc, square_bits=wsq,
                          product_bits=wz, result_bits=wp)
        t0 = time.time()
        try:
            y1 = fixed_point_model(N_P, prec, xs)
            approx = y1 / 2.0**prec.output_bits(N_P)
            rel = np.abs(approx - inv_sqrt) / inv_sqrt
            worst = float(rel.max())
            margin = (2.0**-TARGET_BITS - rel) * inv_sqrt  # in turns
            min_margin = float(margin.min())
            ok = worst <= 2 ** -(TARGET_BITS + FEAS_MARGIN) and min_margin > 0
        except OverflowError:
            worst, ok, min_margin = float("inf"), False, 0.0
        n_rot = wp + 8
        zt = zpow_t_cost(n_rot, min_margin) if ok else float("inf")
        cost_t = toffoli_estimate(prec) * 2  # both directions
        q = qubit_count(prec)
        equiv = cost_t + zt / 4
        results.append((worst, ok, equiv, cost_t, zt, q, (d, wc, wsq, wz, wp)))
        print(f"[{idx+1}/{len(configs)}] d={d} wc={wc} wsq={wsq} wz={wz} wp={wp}: "
              f"worst=2^{math.log2(worst):.2f} {'OK ' if ok else 'FAIL'} "
              f"estT={cost_t:.0f} ZpowT={zt if ok else float('nan'):.0f} "
              f"eqv={equiv:.0f} Q={q} ({time.time()-t0:.1f}s)")

    feas = [r for r in results if r[1]]
    feas.sort(key=lambda r: (r[2] * r[5], r[2]))
    print("\n=== feasible configs by est equiv*Q ===")
    for worst, ok, equiv, cost_t, zt, q, cfg in feas[:25]:
        print(f"  eqv*Q={equiv*q:>9.0f}  eqv={equiv:>7.1f}  estT={cost_t:>6.0f}  "
              f"ZpowT={zt:>5.1f}  Q={q:>3d}  worst=2^{math.log2(worst):.2f}  cfg={cfg}")

    np.savez(Path(__file__).resolve().parent.parent / "results" / "b1_sweep.npz",
             results=np.array([(r[0], r[1], r[2], r[3], r[4], r[5], *r[6]) for r in results],
                              dtype=np.float64))


if __name__ == "__main__":
    main()
