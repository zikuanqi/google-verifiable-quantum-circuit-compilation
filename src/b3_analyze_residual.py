# -*- coding: utf-8 -*-
"""Analyze a b3_work checkpoint: residual sectors, slice costs, E_gap breakdown."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="results/b3_work_*.npz")
    args = ap.parse_args()

    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))

    d = np.load(ROOT / "results" / args.state, allow_pickle=False)
    U = np.array(d["U"]); w = np.array(d["w"]); W = np.array(d["W"])
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    XR = X.reshape(-1, N * N)
    G = XR.T @ XR
    tei = np.asarray(ref_data["tei"], dtype=np.float64)
    T = tei.reshape(N * N, N * N)
    eye = np.eye(N)
    T = T + 0.5 * (S[:, :, None, None] * eye[None, None, :, :]).reshape(N * N, N * N)
    T = T + 0.5 * (eye[:, :, None, None] * S[None, None, :, :]).reshape(N * N, N * N)
    D = G - T
    D4 = D.reshape(N, N, N, N)
    dmask = np.eye(N, dtype=bool)
    omask = ~dmask
    dd = float((D4[dmask][:, dmask] ** 2).sum())
    do = 2 * float((D4[dmask][:, omask] ** 2).sum())
    oo = float((D4[omask][:, omask] ** 2).sum())
    print(f"residual e2^2 = {dd + do + oo:.4f}  (e2={np.sqrt(dd + do + oo):.4f})")
    print(f"  diag-diag: {dd:.4f}   diag-off: {do:.4f}   off-off: {oo:.4f}")
    print("  target sector energies: dd=144.73  do=3.01  oo=0.72")
    a = np.abs(w).sum(axis=1)
    s = w.sum(axis=1)
    xB = W + s
    print(f"a rms={np.sqrt((a**2).mean()):.3f} |wB| rms={np.sqrt((xB**2).mean()):.3f} "
          f"Lam2={0.25 * ((a + np.abs(xB)) ** 2).sum():.2f}")
    Xn = np.linalg.norm(X.reshape(-1, N, N), ord="nuc", axis=(1, 2))
    print(f"slice nuc rms={np.sqrt((Xn**2).mean()):.3f}, a/nuc mean={np.mean(a / np.maximum(Xn, 1e-12)):.2f}")
    h1p = ref_data["h1_thc"] - 0.5 * ref_data["electrons"] * S + beta_1 * np.eye(N) \
        - np.einsum("rc,rcpq->pq", W, X, optimize=True)
    ev = np.linalg.eigvalsh(h1p)
    negsum = -ev[ev < 0].sum()
    esos = ref_data["ecore"] - ref_data["electrons"] * beta_1 - 2 * negsum - 0.5 * (W**2).sum()
    egap = ref.UPPER_ENERGY - esos
    print(f"Lam1={np.abs(ev).sum():.2f} 2sumneg={2*negsum:.2f} 0.5sumW2={0.5*(W**2).sum():.1f} "
          f"Eg={egap:.2f} Lam={np.abs(ev).sum() + 0.25*((a+np.abs(xB))**2).sum():.2f}")


if __name__ == "__main__":
    main()
