# -*- coding: utf-8 -*-
"""Benchmark III projection experiment: eigen-slices -> structured factors.

The unconstrained optimum is x_c = sqrt(mu_c) v_c (top L eigen-slices, error =
spectral tail).  This script measures how much error the STRUCTURED
representation adds:

    x_c ~ (U_rho (khatri-rao) U_rho) w_c   <=>  X_rho c = U_rho diag(w_c) U_rho^T

Alternating projection per group:
  * seed U_rho: top-B right singular vectors of the group's slice matrix,
    reshaped to 76x76, each E_b's dominant eigenvector as u_b;
  * fix U: w_c = ((UoU)^T (UoU))^-1 (UoU)^T x_c   (closed form);
  * fix w: gradient steps on U (unit-row retraction);
then evaluates error_2 of the projected Gram against T_shift.

Usage: python src/b3_project.py --config 16 24 24 --als 400
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


def build_T(ref_data: dict, S: np.ndarray) -> np.ndarray:
    tei = np.asarray(ref_data["tei"], dtype=np.float64)
    T = np.ascontiguousarray(tei.reshape(N * N, N * N))
    eye = np.eye(N)
    T = T + 0.5 * (S[:, :, None, None] * eye[None, None, :, :]).reshape(N * N, N * N)
    T = T + 0.5 * (eye[:, :, None, None] * S[None, None, :, :]).reshape(N * N, N * N)
    return T


def e2_of_X(X: np.ndarray, T: np.ndarray) -> float:
    XR = X.reshape(-1, N * N)
    G = XR.T @ XR
    D = G - T
    return float(np.sqrt((D * D).sum()))


def project_group(xs: np.ndarray, B: int, als_steps: int, lr: float, seed: int) -> np.ndarray:
    """Project C 5776-vectors onto span{(UoU)} with B unit rows; return (B, N) U."""
    C = xs.shape[0]
    rng = np.random.default_rng(seed)
    # seed: top-B right singular vectors of the slice matrix -> reshape -> dom eigvec
    _, _, Vt = np.linalg.svd(xs, full_matrices=False)  # (C, 5776)
    U = np.zeros((B, N))
    for b in range(B):
        E = Vt[b].reshape(N, N)
        E = 0.5 * (E + E.T)
        ev, q = np.linalg.eigh(E)
        U[b] = q[:, np.argmax(np.abs(ev))]
    # U rows unit; also fill any degenerate rows with random directions
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    if B > 0 and np.any(~np.isfinite(U)):
        U = rng.standard_normal((B, N))
        U = U / np.linalg.norm(U, axis=1, keepdims=True)

    def khatri(U_: np.ndarray) -> np.ndarray:
        return np.einsum("bp,bq->pqb", U_, U_).reshape(N * N, B)

    K = khatri(U)
    Gmat = K.T @ K  # (B, B) = (u_b . u_b')^2
    rhs = K.T @ xs.T  # (B, C)
    w = np.linalg.solve(Gmat + 1e-8 * np.eye(B) * (np.trace(Gmat) / B + 1.0), rhs)  # (B, C)
    res = xs.T - K @ w  # (5776, C)
    best_loss = float(np.sum(res * res))
    best_U = U.copy()
    for it in range(als_steps):
        # U gradient: d||r||^2/du_b[i] = -4 sum_c w_bc (R_c u_b)_i ; ascent direction
        Rm = res.reshape(N, N, C)
        M = Rm.transpose(0, 2, 1) @ U.T   # M[p,c,b] = (R_c u_b)[p]
        gU = 4.0 * np.einsum("pcb,bc->pb", M, w)
        gU = gU - np.sum(gU * U.T, axis=0)[None, :] * U.T  # tangent projection
        U = U + lr * gU.T
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        K = khatri(U)
        Gmat = K.T @ K
        rhs = K.T @ xs.T
        w = np.linalg.solve(Gmat + 1e-8 * np.eye(B) * (np.trace(Gmat) / B + 1.0), rhs)
        res = xs.T - K @ w
        loss = float(np.sum(res * res))
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
        if it % 100 == 0:
            print(f"    als it {it}: loss {loss:.6g} (best {best_loss:.6g})")
    K = khatri(best_U)
    Gmat = K.T @ K
    w = np.linalg.solve(Gmat + 1e-8 * np.eye(B) * (np.trace(Gmat) / B + 1.0), K.T @ xs.T)
    return best_U, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, nargs=3, default=[16, 24, 24])
    ap.add_argument("--als", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    R, B, C = args.config
    L = R * C

    t0 = time.time()
    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
    T = build_T(ref_data, S)
    mu = np.load(RESULTS / "b3_spectrum.npz")["mu"]
    Vtop = np.load(RESULTS / "b3_spectrum_vtop.npy", mmap_mode="r")
    print(f"setup in {time.time() - t0:.1f}s; L={L}")

    # eigen-slice reference error (unconstrained optimum)
    xs_all = (Vtop[:, :L] * np.sqrt(mu[:L])[None, :]).T  # (L, 5776)
    tail2 = float((mu[L:] ** 2).sum())
    print(f"eigen-slice tail e2 = {np.sqrt(tail2):.6f}")

    U = np.zeros((R, B, N))
    w = np.zeros((R, B, C))
    t0 = time.time()
    for r in range(R):
        xs = xs_all[r * C:(r + 1) * C]
        Ur, wr = project_group(xs, B, args.als, args.lr, args.seed + r)
        U[r], w[r] = Ur, wr
        print(f"group {r}: done ({time.time() - t0:.1f}s)")
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    e2 = e2_of_X(X, T)
    print(f"\nprojected structured e2 = {e2:.6f}  (eigen-slice tail {np.sqrt(tail2):.6f}, "
          f"budget 0.0165)")
    a = np.abs(w).sum(axis=1)
    print(f"slice cost: sum a^2 = {(a**2).sum():.2f}  (Lambda2 slice part = {0.25*(a**2).sum():.2f})")
    np.savez(RESULTS / f"b3_proj_{R}_{B}_{C}.npz", U=U, w=w, e2=np.array(e2))


if __name__ == "__main__":
    main()
