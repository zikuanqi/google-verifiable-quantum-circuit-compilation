# -*- coding: utf-8 -*-
"""Fix the W-field of a trained/sliced factorization: W = alpha*W_ls with
E_gap = 4.04 root (all h1' eigenvalues >= 0), then report + optionally write a
trainer warm-start npz (U, W, H2_I, H1_I, Hsym_lower) for further polishing.

Usage:
    python src/b3_fix_w.py results/b3_train_65_60_30.npz [--out-warm results/b3_warm_65_60_30.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
RESULTS = ROOT / "results"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76
E_UPPER = ref.UPPER_ENERGY


def load_state(path: Path):
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    if "w" in keys:  # b3_work format
        return np.array(data["U"], dtype=np.float64), np.array(data["w"], dtype=np.float64), \
            np.array(data["W"], dtype=np.float64), None, None
    # trainer format: U, W(weights), H2_I, H1_I, Hsym_lower
    U = np.array(data["U"], dtype=np.float64)
    w = np.array(data["W"], dtype=np.float64)
    W = np.array(data["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    H1_I = np.array(data["H1_I"], dtype=np.float64).reshape(())
    Hsym = np.array(data["Hsym_lower"], dtype=np.float64)
    return U, w, W, H1_I, Hsym


def fix_W(U: np.ndarray, w: np.ndarray, H0: np.ndarray, eg_target: float = 4.04) -> np.ndarray:
    """W = alpha * w_ls, w_ls = argmin ||sum W X - H0||, alpha = E_gap root."""
    R, C = w.shape[0], w.shape[2]
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = X.reshape(R * C, N, N)
    L = R * C
    Gll = np.einsum("cpq,dpq->cd", X, X)
    rhs = np.einsum("cpq,pq->c", X, H0)
    wls = np.linalg.solve(Gll + 1e-6 * np.eye(L) * (np.trace(Gll) / L + 1.0), rhs)
    SX = np.einsum("c,cpq->pq", wls, X)
    wls2 = float(np.sum(wls * wls))

    def egap_of(alpha: float) -> float:
        ev = np.linalg.eigvalsh(H0 - alpha * SX)
        neg = float(-ev[ev < 0].sum())
        return -559.132 + 2.0 * neg + 0.5 * alpha * alpha * wls2

    alpha0 = np.sqrt(1126.34 / max(wls2, 1e-9))
    if egap_of(alpha0) <= 4.5:
        alpha = alpha0
    else:
        grid = np.linspace(0.0, alpha0, 80)
        egs = np.array([egap_of(a) for a in grid])
        idx = np.where(egs <= eg_target)[0]
        if len(idx) == 0:
            alpha = alpha0
        else:
            k = idx[-1]
            lo, hi = float(grid[k]), float(grid[k + 1])
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if egap_of(mid) >= eg_target:
                    hi = mid
                else:
                    lo = mid
            alpha = 0.5 * (lo + hi)
    return (wls * alpha).reshape(R, C)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out-warm", default=None, help="write trainer warm-start npz")
    ap.add_argument("--eg-target", type=float, default=4.04)
    args = ap.parse_args()

    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    H0 = ref_data["h1_thc"] - 0.5 * ref_data["electrons"] * S + beta_1 * np.eye(N)

    U, w, W_old, H1_I, Hsym = load_state(RESULTS / args.state)
    print(f"state {args.state}: U {U.shape}, w {w.shape}")
    W_new = fix_W(U, w, H0, args.eg_target)

    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    def metrics(W_):
        h1p = dm.build_h1_prime(ref_data["h1_thc"], S, beta_1, ref_data["electrons"], W_, X)
        _, evecs = np.linalg.eigh(h1p)
        u_all = np.vstack([evecs.T, U.reshape(-1, U.shape[-1])])
        return dm.evaluate_dfthc_vectors(u_all, w, W_, S, beta_1, ref_data,
                                         energy_upper=E_UPPER)

    print("\n== old W (from file) ==")
    m = metrics(W_old)
    print(f"  e2={m['error_2']:.6f} Lam={m['Lambda']:.3f} Eg={m['E_gap']:.4f} leff={m['lambda_eff']:.4f}")
    print("== fixed W ==")
    m = metrics(W_new)
    print(f"  e2={m['error_2']:.6f} Lam={m['Lambda']:.3f} Eg={m['E_gap']:.4f} leff={m['lambda_eff']:.4f}")
    a = np.abs(w).sum(axis=1)
    xB = W_new + w.sum(axis=1)
    print(f"  Lam2={0.25*((a+np.abs(xB))**2).sum():.3f}  sumW2={float(np.sum(W_new**2)):.1f}")

    if args.out_warm:
        H2_I = W_new + w.sum(axis=1)
        np.savez(RESULTS / args.out_warm,
                 U=U.astype(np.float32), W=w.astype(np.float32), H2_I=H2_I.astype(np.float32),
                 H1_I=shifts["H1_I"].astype(np.float32), Hsym_lower=shifts["Hsym_lower"].astype(np.float32))
        print(f"wrote warm-start {args.out_warm}")
    np.savez(RESULTS / "b3_work_65_60_30_fixedw.npz", U=U, w=w, W=W_new)
    print("wrote results/b3_work_65_60_30_fixedw.npz (b3_work format)")


if __name__ == "__main__":
    main()
