# -*- coding: utf-8 -*-
"""Benchmark III diagnostics: baseline slice structure, W=0 physics, eigen-group structure."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76


def load_baseline():
    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                   allow_pickle=False)
    w = np.asarray(fact["W"], dtype=np.float64)
    W = np.asarray(fact["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    U = np.array(fact["U"], dtype=np.float64)
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    return ref_data, S, beta_1, w, W, U, X


def main() -> None:
    t0 = time.time()
    ref_data, S, beta_1, w, W, U, X = load_baseline()
    R, B, C = w.shape
    print(f"loaded baseline in {time.time()-t0:.1f}s  (R,B,C)=({R},{B},{C})")

    # --- slice-level costs ---
    a = np.abs(w).sum(axis=1)                      # (R,C) sum_b |w_b|
    wB = W + w.sum(axis=1)
    cost = (a + np.abs(wB)) ** 2
    lam2 = 0.25 * cost.sum()
    print(f"\n== baseline slice structure ==")
    print(f"  Lambda2 = {lam2:.4f}")
    print(f"  per-block (a+|w_B|): rms={np.sqrt(cost.mean()):.4f} "
          f"a rms={np.sqrt((a**2).mean()):.4f} |w_B| rms={np.sqrt((wB**2).mean()):.4f}")
    Xf = np.linalg.norm(X, axis=(2, 3))            # (R,C) ||X_rc||_F
    Xn = np.linalg.norm(X.reshape(R * C, N, N), ord="nuc", axis=(1, 2)).reshape(R, C)
    print(f"  ||X_rc||_F : rms={np.sqrt((Xf**2).mean()):.4f} sum f^2 = {(Xf**2).sum():.4f}")
    print(f"  ||X_rc||_* : rms={np.sqrt((Xn**2).mean()):.4f} sum nuc^2 = {(Xn**2).sum():.4f}")
    print(f"  a vs nuc ratio (excess LCU cost): mean={np.mean(a / np.maximum(Xn, 1e-12)):.3f} "
          f"median={np.median(a / np.maximum(Xn, 1e-12)):.3f}")
    lam2_min_repr = 0.25 * (Xn**2).sum()  # if every slice re-represented in its own eigenbasis
    print(f"  Lambda2 lower bound (per-slice eigen repr): {lam2_min_repr:.4f}")

    # --- W=0 physics ---
    h1_thc = ref_data["h1_thc"]
    h1p0 = h1_thc - 0.5 * ref_data["electrons"] * S + beta_1 * np.eye(N)
    ev0 = np.linalg.eigvalsh(h1p0)
    lam1_0 = np.abs(ev0).sum()
    negsum0 = -ev0[ev0 < 0].sum()
    esos0 = ref_data["ecore"] - ref_data["electrons"] * beta_1 - 2 * negsum0
    egap0 = ref.UPPER_ENERGY - esos0
    print(f"\n== W = 0 physics (X,w irrelevant) ==")
    print(f"  h1' = H1_THC - eta/2 S + beta1 I: evals [{ev0.min():.4f},{ev0.max():.4f}]")
    print(f"  Lambda1(W=0) = {lam1_0:.4f}, 2*sum|neg| = {2*negsum0:.4f}")
    print(f"  E_SOS(W=0) = {esos0:.6f}  -> E_gap(W=0) = {egap0:.6f}  (floor 3.914898)")
    # baseline actual
    h1p = h1_thc - 0.5 * ref_data["electrons"] * S + beta_1 * np.eye(N) \
        - np.einsum("rc,rcpq->pq", W, X, optimize=True)
    ev = np.linalg.eigvalsh(h1p)
    lam1 = np.abs(ev).sum()
    negsum = -ev[ev < 0].sum()
    print(f"  baseline: Lambda1={lam1:.4f}, 2*sum|neg|={2*negsum:.4f}, 0.5*sum W^2={0.5*(W**2).sum():.4f}")

    # --- eigen-group structure: how commutable are the top eigen-matrices? ---
    print("\n== top eigen-matrix group structure ==")
    mu = np.load(ROOT / "results" / "b3_spectrum.npz")["mu"]
    Vt = np.load(ROOT / "results" / "b3_spectrum_vtop.npy", mmap_mode="r")
    k = min(384, Vt.shape[1])
    Vm = np.asarray(Vt[:, :k]).T.reshape(k, N, N)
    # symmetrize (should be symmetric already)
    Vm = 0.5 * (Vm + Vm.transpose(0, 2, 1))
    print(f"  symmetry check: max|V - V^T| = {np.abs(Vm - Vm.transpose(0, 2, 1)).max():.2e}")
    # top-singular-vector rank structure
    sv = np.linalg.svd(Vm, compute_uv=False)
    top1 = (sv[:, 0] ** 2) / np.maximum((sv**2).sum(axis=1), 1e-300)
    print(f"  top-{k} eigen-matrices: frac energy in top singular vec: "
          f"mean={top1.mean():.3f} median={np.median(top1):.3f} min={top1.min():.3f}")
    # commutator norms of the top 24 (sample)
    pairs = np.zeros(24)
    n = 0
    for i in range(23):
        Ci = Vm[i] @ Vm[i + 1] - Vm[i + 1] @ Vm[i]
        pairs[i] = np.linalg.norm(Ci) / (np.linalg.norm(Vm[i]) * np.linalg.norm(Vm[i + 1]) + 1e-30)
    print(f"  adjacent commutator ||[V_j, V_j+1]||/(||V_j||||V_j+1||): "
          f"mean={pairs.mean():.3f} max={pairs.max():.3f} (0 = commute)")
    # diagonal dominance in the canonical (orbital) basis
    diag_frac = (np.diagonal(Vm, axis1=1, axis2=2) ** 2).sum(axis=1) / (Vm**2).sum(axis=(1, 2))
    print(f"  diag-energy fraction in orbital basis: mean={diag_frac.mean():.3f} "
          f"(first20: {np.round(diag_frac[:20], 2)})")

    # --- what error does a rank-L eigen-slice fit give with joint diag in ONE basis? ---
    # try the joint diagonalizer of the top-k V_j via sum mu_j V_j^2 eigenspace
    A = np.einsum("jpq,jpr->qr", mu[:k, None, None] * Vm, Vm)
    _, Ub = np.linalg.eigh(A)  # columns
    off = np.zeros(k)
    for j in range(k):
        Vd = Ub.T @ Vm[j] @ Ub
        off[j] = (Vd**2).sum() - (np.diag(Vd) ** 2).sum()
    err1 = np.sqrt((mu[:k] * off).sum())
    print(f"  single joint basis (eig of sum mu V^2): err2 est from off-diag = {err1:.6f}")
    # and tail
    tail = np.sqrt((mu[k:]**2).sum())
    print(f"  tail beyond {k}: {tail:.6f} -> combined {np.hypot(err1, tail):.6f}")
    # diag fraction profile
    for j0, j1 in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 384)]:
        if j0 < k:
            j1 = min(j1, k)
            print(f"  diag-fraction j in [{j0},{j1}): mean={diag_frac[j0:j1].mean():.3f} "
                  f"min={diag_frac[j0:j1].min():.3f}")

    # --- one-body consistency: contraction of T vs -H0 ---
    print("\n== one-body consistency ==")
    tei = ref_data["tei"]
    h1_thc = ref_data["h1_thc"]
    eta = ref_data["electrons"]
    CT = np.einsum("prrq->pq", tei) + 0.5 * N * S + 0.5 * np.trace(S) * np.eye(N)
    H0 = h1_thc - 0.5 * eta * S + beta_1 * np.eye(N)
    print(f"  C_T = contraction of T_shift (evals): {np.round(np.linalg.eigvalsh(CT), 3)}")
    print(f"  -H0 (evals): {np.round(np.linalg.eigvalsh(-H0), 3)}")
    D = CT + H0
    print(f"  C_T - (-H0) = C_T + H0: ||.||_F = {np.linalg.norm(D):.4f}, evals: {np.round(np.linalg.eigvalsh(D), 3)}")
    print(f"  tr(C_T) = {np.trace(CT):.4f}, tr(-H0) = {-np.trace(H0):.4f}")

    # --- pair-sector energy split of T_shift (full N^4 Frobenius) ---
    print("\n== pair-sector split of T_shift ==")
    T = np.asarray(tei, dtype=np.float64)  # (p,q,s,t)
    T = T + 0.5 * S[:, :, None, None] * np.eye(N)[None, None, :, :]
    T = T + 0.5 * np.eye(N)[:, :, None, None] * S[None, None, :, :]
    tot2 = float((T**2).sum())
    print(f"  ||T||_F^2 = {tot2:.4f}  (sqrt = {np.sqrt(tot2):.4f}), from spectrum: sum mu^2 = {(mu**2).sum():.4f}")
    diag_mask = np.eye(N, dtype=bool)
    off_mask = ~diag_mask
    # diag-diag block: T[p,p,s,s]
    dd2 = float((T[diag_mask][:, diag_mask] ** 2).sum())
    # diag-offdiag blocks: (p==q, s!=t) and (p!=q, s==t): equal by symmetry
    do2 = float((T[diag_mask][:, off_mask] ** 2).sum()) * 2.0
    # off-off block: p!=q, s!=t
    oo2 = tot2 - dd2 - do2
    print(f"  diag-diag: {dd2:.4f} ({100*dd2/tot2:.1f}%)   diag-off: {do2:.4f} ({100*do2/tot2:.1f}%)   "
          f"off-off: {oo2:.4f} ({100*oo2/tot2:.1f}%)")
    # Coulomb block C[p,s] = T[p,p,s,s]
    Cm = np.array(T[diag_mask][:, diag_mask])
    evC = np.linalg.eigvalsh(Cm)
    print(f"  Coulomb 76x76: evals [{evC.min():.3f},{evC.max():.3f}], "
          f"sum={evC.sum():.3f}, neg part {evC[evC < 0].sum():.3f}")
    # exchange: the off-diag pair block with 8-fold multiplicity
    # tei[p,q,s,t] for p<q, s<t: build the 2850x2850 matrix once (dense)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    idx = {(p, q): k for k, (p, q) in enumerate(pairs)}
    Bx = np.zeros((len(pairs), len(pairs)))
    for k, (p, q) in enumerate(pairs):
        for l, (s, t) in enumerate(pairs):
            Bx[k, l] = tei[p, q, s, t]
    evB = np.linalg.eigvalsh(Bx)
    print(f"  exchange 2850x2850 (off-diag pairs, tei values): evals [{evB.min():.3f},{evB.max():.3f}], "
          f"||B||^2*4={4*float((Bx**2).sum()):.4f} (vs off-off {oo2:.4f}), "
          f"sum evals={evB.sum():.3f}")
    # ranks of top exchange eigenmatrices
    order = np.argsort(np.abs(evB))[::-1]
    topB = np.abs(evB[order])
    print(f"  top |exchange evals|: {np.round(topB[:12], 3)}")
    # singular profile of the leading exchange eigenvectors as matrices
    _, VB = np.linalg.eigh(Bx)
    VB = VB[:, order]
    for j in (0, 4, 9, 19, 49):
        E = np.zeros((N, N))
        for k, (p, q) in enumerate(pairs):
            E[p, q] = VB[k, j]
            E[q, p] = VB[k, j]
        svE = np.linalg.svd(E, compute_uv=False)
        c2 = np.cumsum(svE**2)
        r90 = int(np.searchsorted(c2, 0.9 * c2[-1]) + 1)
        print(f"    exchange eig j={j:3d} (|ev|={topB[j]:.4f}): matrix rank90={r90}, "
              f"||E||_*={svE.sum():.3f}, tr={np.trace(E):.3f}")


if __name__ == "__main__":
    main()
