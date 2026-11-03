# -*- coding: utf-8 -*-
"""B3: scale W of a (U, w, W) checkpoint so that E_gap ~= target, keeping e2
(which depends only on X from U and w) exactly unchanged. Report all metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "verifiers"))
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402
from dfthcsos.scoreinfra import reference as ref  # noqa: E402


def metrics(U, w, W, ref_data, S, beta_1):
    X = dm.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1p = dm.build_h1_prime(ref_data["h1_thc"], S, beta_1, ref_data["electrons"], W, X)
    evals = np.linalg.eigvalsh(h1p)
    lam1 = float(np.sum(np.abs(evals)))
    tr = float(np.sum(evals))
    lam2 = dm.lambda_h2(w, W)
    Lambda = lam1 + lam2
    E_SOS = (ref_data["ecore"] - ref_data["electrons"] * beta_1
             - (lam1 - tr) - 0.5 * np.sum(W**2))
    E_gap = ref.UPPER_ENERGY - E_SOS
    in_dom = 0.0 < E_gap < 2 * Lambda
    leff = float(np.sqrt(E_gap * (2 * Lambda - E_gap))) if in_dom else float("nan")
    h2 = dm.reconstruct_h2(X) - ref_data["tei"]
    diag = np.arange(76)
    h2[:, :, diag, diag] -= 0.5 * S[:, :, None]
    h2[diag, diag, :, :] -= 0.5 * S[None, :, :]
    e2 = float(np.linalg.norm(h2.ravel()))
    return dict(e2=e2, Lambda=Lambda, lam1=lam1, lam2=lam2, E_gap=E_gap,
                lambda_eff=leff, in_dom=in_dom, sumW2=float(np.sum(W**2)),
                neg_part=float(lam1 - tr))


def main():
    ckpt = np.load(sys.argv[1] if len(sys.argv) > 1 else
                   str(ROOT / "results" / "b3_work_65_60_30_opt_ckpt.npz"))
    U, w, W = (np.asarray(ckpt[k], dtype=np.float64) for k in ("U", "w", "W"))
    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = dm.symmetric_from_lower(shifts["Hsym_lower"], 76).astype(np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))

    m0 = metrics(U, w, W, ref_data, S, beta_1)
    print("as-is:", {k: round(v, 6) if isinstance(v, float) else v for k, v in m0.items()})

    # Need: E_gap = E_upper - E_SOS - 0.5*sumW2_new + 0.5*sumW2_old ... but the
    # h1' spectrum also changes with W. Solve by bisection on a scale factor s.
    target_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 4.04
    lo, hi = 0.0, 1.0
    # find a scale with gap <= target at hi by growing hi
    for _ in range(60):
        m = metrics(U, w, W * hi, ref_data, S, beta_1)
        if m["E_gap"] <= target_gap:
            break
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = metrics(U, w, W * mid, ref_data, S, beta_1)
        if m["E_gap"] > target_gap:
            lo = mid
        else:
            hi = mid
    s = hi
    m1 = metrics(U, w, W * s, ref_data, S, beta_1)
    print(f"scaled W by {s:.6f} (sumW2 {m0['sumW2']:.3f} -> {m1['sumW2']:.3f}):")
    print({k: round(v, 6) if isinstance(v, float) else v for k, v in m1.items()})
    # also try a few targets
    for tg in (3.92, 3.95, 4.0, 4.1, 4.5, 5.0):
        lo, hi = 0.0, s * 2 + 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            mm = metrics(U, w, W * mid, ref_data, S, beta_1)
            if mm["E_gap"] > tg:
                lo = mid
            else:
                hi = mid
        mm = metrics(U, w, W * hi, ref_data, S, beta_1)
        print(f"target gap {tg}: scale {hi:.5f} lambda_eff={mm['lambda_eff']:.4f} "
              f"Lambda={mm['Lambda']:.3f} E_gap={mm['E_gap']:.4f}")


if __name__ == "__main__":
    main()
