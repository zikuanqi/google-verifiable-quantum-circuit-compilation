# -*- coding: utf-8 -*-
"""Benchmark III (DFTHC-SOS): full spectrum of T_shift and rank/error analysis.

T_shift[(p,q),(r,s)] = tei[p,q,r,s] + 0.5*S[p,r]*[q==s] + 0.5*[p==r]*S[q,s]
is the exact two-body target of error_2:  the submitted Gram tensor
sum_rc X^rc (x) X^rc is PSD on the pair space, so every negative eigenvalue of
T_shift is unavoidable error, and a rank-L PSD approximation misses the tail of
positive eigenvalues beyond L.

This script:
  * eigendecomposes the 5776 x 5776 float64 matrix (MKL, a few minutes),
  * saves eigenvalues + top eigenvectors to results/b3_spectrum.npz,
  * prints the error floor, tail(L) curves, and the LCU-cost structure
    (nuclear norms of the top eigen-matrices) that the optimizer needs.

Usage:  python src/b3_spectrum.py [--force] [--top K]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
RESULTS = ROOT / "results"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76
OUT = RESULTS / "b3_spectrum.npz"
VTOP_FILE = RESULTS / "b3_spectrum_vtop.npy"


def build_tshift(tei: np.ndarray, S: np.ndarray) -> np.ndarray:
    """T_shift as a (N^2, N^2) float64 matrix (tei must be float32-rounded values).

    Matches the verifier's code exactly (dfthc_math.evaluate_dfthc_vectors):
      h2_delta[p,q,s,t] = G[p,q,s,t] - tei[p,q,s,t]
      h2_delta[:, :, d, d]  -= 0.5 * S[:, :]        (0.5*S[p,q] on s==t)
      h2_delta[d, d, :, :]  -= 0.5 * S[:, :]        (0.5*S[s,t] on p==q)
    so T_shift[p,q,s,t] = tei + 0.5*S[p,q]*[s==t] + 0.5*[p==q]*S[s,t].
    """
    t0 = time.time()
    T = np.ascontiguousarray(np.asarray(tei, dtype=np.float64).reshape(N * N, N * N))
    eye = np.eye(N, dtype=np.float64)
    # 0.5 * S[p,q] * [s == t]
    T += 0.5 * (S[:, :, None, None] * eye[None, None, :, :]).reshape(N * N, N * N)
    # 0.5 * [p == q] * S[s, t]
    T += 0.5 * (eye[:, :, None, None] * S[None, None, :, :]).reshape(N * N, N * N)
    print(f"[spectrum] T_shift built in {time.time() - t0:.1f}s, max|T-Tt|={np.abs(T - T.T).max():.3e}")
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    ap.add_argument("--top", type=int, default=2200, help="eigenvectors saved (columns)")
    ap.add_argument("--no-save", action="store_true", help="do not write outputs")
    args = ap.parse_args()
    d2 = N * N

    RESULTS.mkdir(exist_ok=True)
    if not args.no_save and OUT.exists() and not args.force:
        data = np.load(OUT, allow_pickle=False)
        mu = np.array(data["mu"])
        print(f"[spectrum] loaded {OUT} (mu shape {mu.shape})")
    else:
        t0 = time.time()
        ref_data = ref.load_reference(ref.check_fcidump())
        shifts = ref.load_fixed_shifts()
        S = np.asarray(
            dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64
        )
        print(f"[spectrum] reference loaded in {time.time() - t0:.1f}s")

        T = build_tshift(ref_data["tei"], S)
        # cross-check: ||G_baseline - T_shift||_F must equal the verifier's 0.016133
        fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                       allow_pickle=False)
        w = np.asarray(fact["W"], dtype=np.float64)
        W = np.asarray(fact["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
        Ub = np.array(fact["U"], dtype=np.float64)
        Ub = Ub / np.linalg.norm(Ub, axis=2, keepdims=True)
        Xb = np.einsum("tjp,tjq,tjl->tlpq", Ub, Ub, w, optimize=True)
        Xb = np.tril(Xb) + np.swapaxes(np.tril(Xb, k=-1), -1, -2)
        XR = Xb.reshape(Xb.shape[0] * Xb.shape[1], d2)
        G = XR.T @ XR
        print(f"[spectrum] cross-check ||G_baseline - T_shift||_F = {np.linalg.norm(G - T):.8f} "
              f"(verifier: 0.016133)")
        del ref_data, fact, w, W, Ub, Xb, XR, G

        t0 = time.time()
        mu_asc, V = np.linalg.eigh(T)  # ascending
        print(f"[spectrum] eigh done in {time.time() - t0:.1f}s")
        del T
        order = np.argsort(mu_asc)[::-1]
        mu = mu_asc[order]
        V = V[:, order]

        if not args.no_save:
            t0 = time.time()
            np.savez(OUT, mu=mu, args_top=args.top)
            vtop_out = VTOP_FILE if args.top <= 1600 else RESULTS / "b3_spectrum_vtop2.npy"
            np.save(vtop_out, V[:, : args.top])
            print(f"[spectrum] saved mu + Vtop({args.top}) in {time.time() - t0:.1f}s")

    # ---------------- diagnostics ----------------
    d2 = N * N
    neg = mu[mu < 0.0]
    pos = mu[mu > 0.0]
    floor2 = float(np.sum(neg**2))
    floor = np.sqrt(floor2)
    print(f"\n== spectrum of T_shift ({d2} x {d2}) ==")
    print(f"  mu range: [{mu.min():.6g}, {mu.max():.6g}], sum mu = {mu.sum():.6g}")
    print(f"  positives: {pos.size}, negatives: {neg.size}, |mu|<1e-12: {int(np.sum(np.abs(mu) < 1e-12))}")
    print(f"  negative sum = {neg.sum():.6g}, neg part of ||T||: sqrt(sum neg mu^2) = {floor:.6f}")
    cs = np.sqrt(np.cumsum(pos[::-1] ** 2))[::-1]  # cs[k] = sqrt(sum_{j>=k} mu_j^2), j over pos
    print(f"  achievable error_2 lower bound vs rank L (pos tail + neg floor):")
    for L in (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2926, 3900):
        tail_pos = cs[L] if L < pos.size else 0.0
        print(f"    L={L:5d}: total >= {np.hypot(tail_pos, floor):.6f}   (pos tail {tail_pos:.6f})")

    # ---- eigen-matrix structure of the top eigenvectors (LCU-cost predictors) ----
    print("\n== top eigen-matrices (reshaped 76x76): rank / nuclear norm / trace ==")
    print("  (X_rc = sqrt(mu) V_j costs 0.25 * mu * ||V_j||_*^2 in Lambda2 with an eigen rep)")
    ktop = min(800, args.top)
    Vt = np.load(VTOP_FILE, mmap_mode="r") if VTOP_FILE.exists() else None
    if Vt is not None:
        ktop = min(ktop, Vt.shape[1])
        Vm = np.asarray(Vt[:, :ktop]).T.reshape(ktop, N, N)  # (ktop, 76, 76)
        mu_k = mu[:ktop]
        # nuclear norm via SVD of each 76x76 (thin)
        t0 = time.time()
        sv = np.linalg.svd(Vm, compute_uv=False)  # (ktop, 76)
        print(f"  svd of {ktop} eigen-matrices in {time.time() - t0:.1f}s")
        nuc = sv.sum(axis=1)
        r1 = (sv[:, 0] / np.sqrt(np.maximum((sv**2).sum(axis=1), 1e-300)))  # top sing / frob
        trs = np.trace(Vm, axis1=1, axis2=2)
        for name, sl in [("all", slice(0, ktop)), ("first100", slice(0, 100))]:
            m = mu_k[sl]
            nn = nuc[sl]
            tt = trs[sl]
            r1s = r1[sl]
            print(f"  [{name}] n={m.size}: mu in [{m.min():.4g},{m.max():.4g}], sum mu={m.sum():.4g}")
            print(f"    ||V||_* : mean={nn.mean():.3f} median={np.median(nn):.3f} max={nn.max():.3f}")
            print(f"    top-sing/frob: mean={r1s.mean():.3f} median={np.median(r1s):.3f}")
            print(f"    tr(V): mean={tt.mean():.4f} rms={np.sqrt((tt**2).mean()):.4f} max|tr|={np.abs(tt).max():.4f}")
            print(f"    Lambda2 est (eigen-slice, w_B=0): 0.25*sum mu*||V||_*^2 = {0.25*float((m*nn**2).sum()):.6g}")
            print(f"    Lambda2 est (rank-1 proj): 0.25*sum mu = {0.25*float(m.sum()):.6g}")
            print(f"    1/2 sum tr(X)^2 est (E_SOS, W=-s): 0.5*sum mu*tr(V)^2 = {0.5*float((m*tt**2).sum()):.6g}")
        # rank profile of the top 20
        print("  top-20 eigen-matrices: mu, rank80/rank95/rank99, ||V||_*, tr(V)")
        for j in range(20):
            s = sv[j]
            c2 = np.cumsum(s**2)
            r80 = int(np.searchsorted(c2, 0.8 * c2[-1]) + 1)
            r95 = int(np.searchsorted(c2, 0.95 * c2[-1]) + 1)
            r99 = int(np.searchsorted(c2, 0.99 * c2[-1]) + 1)
            print(f"    j={j:3d} mu={mu[j]:10.5g} rank(80/95/99%)={r80}/{r95}/{r99} "
                  f"||V||_*={nuc[j]:6.3f} tr={trs[j]:9.5f}")

    # ---- the (R,B,C) sweep against the achievable error ----
    print("\n== (R, C) candidates: rank L=R*C, B=C assumed, tail error ==")
    for R, B, C in [(96, 8, 8), (48, 16, 16), (32, 24, 24), (24, 32, 32),
                    (20, 40, 40), (32, 32, 32), (64, 16, 16), (128, 8, 8),
                    (65, 60, 60), (40, 40, 40), (16, 24, 24), (10, 40, 40)]:
        L = R * C
        tail_pos = cs[L] if L < pos.size else 0.0
        t = np.hypot(tail_pos, floor)
        print(f"  (R,B,C)=({R:3d},{B:3d},{C:3d})  L={L:5d}  rows={76 + R * B:5d}  "
              f"unconstrained tail error >= {t:.6f}")


if __name__ == "__main__":
    main()
