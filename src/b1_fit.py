# -*- coding: utf-8 -*-
"""Benchmark I v2: offline best-fit polynomial coefficients per dyadic interval.

Fits c(x) = c0 - c1*d + c2*d^2 (- c3*d^3) to 1/sqrt(t) on each dyadic interval
with minimax (L-infinity) error, as a linear program, then quantizes to `wc`
bits and reports the worst-case RELATIVE error of y0 = poly(M) over the
interval, plus the worst-case error after ONE Newton-Raphson step
(rel_err_after_NR ~= rel_err^2 * M/2 + truncation terms).

The true post-NR error (with fixed-point truncations) is checked later by the
full integer model (b1_v2_model); this module answers the question: which
(K, D) choices can reach 2^-24 after one NR step.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog


def fit_poly_interval(lo: float, hi: float, degree: int, n: int = 4000) -> np.ndarray:
    """Minimax polynomial fit of 1/sqrt(t) on [lo, hi] with monomial basis
    [1, -(t-lo), (t-lo)^2, -(t-lo)^3], degree `degree` (1..3).

    Uses an LP over a fine grid with a post-solve refinement pass:
    minimize eps s.t. |P(t) - f(t)| * sqrt(t) <= eps  (relative error).
    Returns coefficients (c0..c_degree) for the alternating-sign basis.
    """
    t = np.linspace(lo, hi, n)
    f = 1.0 / np.sqrt(t)
    basis = [t**0]
    for k in range(1, degree + 1):
        basis.append(-((t - lo) ** k) if k % 2 == 1 else (t - lo) ** k)
    A = np.column_stack(basis)
    w = np.sqrt(t)  # scale so that |poly - f|*sqrt(t) <= eps
    Aw = A * w[:, None]
    fw = f * w
    # LP: minimize eps; |Aw c - fw| <= eps
    m = A.shape[1]
    c = np.zeros(m + 1)
    c[-1] = 1.0
    A_ub = np.block([[Aw, -np.ones((n, 1))], [-Aw, -np.ones((n, 1))]])
    b_ub = np.concatenate([fw, -fw])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * m + [(0, None)],
                  method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed on [{lo}, {hi}]: {res.message}")
    return res.x[:-1], res.x[-1]


def quantize_and_eval(coefs: np.ndarray, lo: float, hi: float, degree: int, wc: int,
                      n: int = 20000) -> tuple[float, float]:
    """Round coefficients to wc fractional bits; return worst |rel err| of y0 and
    the predicted worst |rel err| after one NR step (err^2 * t/2)."""
    cq = np.round(coefs * 2**wc).astype(np.int64)
    t = np.linspace(lo, hi, n)
    d = t - lo
    p = np.full_like(t, float(cq[0]))
    for k in range(1, degree + 1):
        term = (-(d**k) if k % 2 == 1 else d**k) * cq[k]
        p = p + term
    p = p / 2**wc
    rel = np.abs(p - 1.0 / np.sqrt(t)) * np.sqrt(t)
    e0 = float(rel.max())
    # NR: y1 = y0*(3 - t*y0^2)/2; relative error ~ (rel^2) * t / 2 approximately.
    enr = float(((rel**2) * t / 2).max())
    return e0, enr


def main():
    print("K intervals x degree D: worst y0 rel err -> predicted post-NR rel err (2^-24.4 needed)")
    for K in (4, 8, 16, 32, 64):
        h = 4.0 / K
        for D in (1, 2, 3):
            worst_e0, worst_enr = 0.0, 0.0
            ok = True
            for i in range(K):
                lo, hi = 1.0 + i * h, 1.0 + (i + 1) * h
                try:
                    coefs, _ = fit_poly_interval(lo, hi, D)
                    e0, enr = quantize_and_eval(coefs, lo, hi, D, 17)
                except RuntimeError as e:
                    ok = False
                    break
                worst_e0 = max(worst_e0, e0)
                worst_enr = max(worst_enr, enr)
            if not ok:
                print(f"  K={K:3d} D={D}: LP failed")
                continue
            status = "OK " if worst_enr <= 2**-24.4 else "    "
            print(f"  K={K:3d} D={D}: y0err=2^{math.log2(worst_e0):6.2f}  "
                  f"NRerr~2^{math.log2(worst_enr):6.2f}  {status}")


if __name__ == "__main__":
    main()
