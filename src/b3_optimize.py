# -*- coding: utf-8 -*-
"""Benchmark III (DFTHC-SOS) factorization optimizer.

Minimizes  lambda_eff^2 + error/E_gap penalties  over (U, w, W) for a fixed
(R, B, C) shape:

    X_rc = sum_b w_b,rc * u_b^rho u_b^rho^T     (u rows unit norm)
    Lambda2 = 1/4 sum_rc (sum_b |w_b| + |W_rc + sum_b w_b|)^2
    h1' = H1_THC - eta/2 S + beta1 I - sum_rc W_rc X_rc
    Lambda1 = sum |eig(h1')|,  neg = sum_{lam<0} |lam|
    E_SOS = ecore - eta*beta1 - 2*neg - 1/2 sum W^2
    E_gap = E_upper - E_SOS,   lambda_eff = sqrt(E_gap * (2(Lambda1+Lambda2) - E_gap))
    error_2 = ||sum_rc X_rc (x) X_rc - T_shift||_F

All gradients are analytic (finite-difference verified with --check-grads);
Adam on (U, w, W) with unit-row retraction for U.

Usage:
    python src/b3_optimize.py --config 16 24 24 --steps 400 --lr 3e-3
    python src/b3_optimize.py --check-grads
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
E_UPPER = ref.UPPER_ENERGY


# ---------------------------------------------------------------------------
# reference data + T_shift
# ---------------------------------------------------------------------------

class Problem:
    """Fixed problem data: reference integrals, shifts, T_shift, spectrum."""

    def __init__(self, top: int = 1600) -> None:
        t0 = time.time()
        self.ref_data = ref.load_reference(ref.check_fcidump())
        shifts = ref.load_fixed_shifts()
        self.S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
        self.beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(self.S))
        self.h1_thc = np.asarray(self.ref_data["h1_thc"], dtype=np.float64)
        self.H0 = self.h1_thc - 0.5 * self.ref_data["electrons"] * self.S \
            + self.beta_1 * np.eye(N)
        self.ecore = float(self.ref_data["ecore"])
        self.eta = int(self.ref_data["electrons"])
        tei = np.asarray(self.ref_data["tei"], dtype=np.float64)
        T = tei.reshape(N * N, N * N)
        eye = np.eye(N)
        T = T + 0.5 * (self.S[:, :, None, None] * eye[None, None, :, :]).reshape(N * N, N * N)
        T = T + 0.5 * (eye[:, :, None, None] * self.S[None, None, :, :]).reshape(N * N, N * N)
        self.T = np.ascontiguousarray(T)
        self.Tsq = float((T * T).sum())
        print(f"[opt] problem built in {time.time() - t0:.1f}s")
        try:
            spec = np.load(RESULTS / "b3_spectrum.npz", allow_pickle=False)
            self.mu = np.asarray(spec["mu"])
            vtop_files = [RESULTS / "b3_spectrum_vtop2.npy", RESULTS / "b3_spectrum_vtop.npy"]
            existing = [f for f in vtop_files if f.is_file()]
            best = None
            for f in existing:
                arr = np.load(f, mmap_mode="r")
                if best is None or arr.shape[1] > best.shape[1]:
                    best = arr
            self.Vtop = best
            self.ktop = int(self.Vtop.shape[1]) if self.Vtop is not None else 0
            print(f"[opt] spectrum loaded: mu shape {self.mu.shape}, Vtop cols {self.ktop}")
        except OSError:
            print("[opt] WARNING: no spectrum files; run src/b3_spectrum.py first")
            self.mu = None
            self.Vtop = None
            self.ktop = 0


# ---------------------------------------------------------------------------
# forward pass + gradients
# ---------------------------------------------------------------------------

def _safe_sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def forward(prob: Problem, U: np.ndarray, w: np.ndarray, W: np.ndarray) -> dict:
    """All quantities and gradients; returns dict with 'phi' plus grad arrays."""
    R, B, C = w.shape
    # normalize U rows (unit norm), X
    Un = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", Un, Un, w, optimize=True)   # (R,C,N,N) symmetric
    XR = X.reshape(R * C, N * N)
    G = XR.T @ XR                                                  # (N^2, N^2) Gram
    D = G - prob.T
    e2sq = float((D * D).sum())
    e2 = np.sqrt(e2sq)

    # Lambda2 and its w/W gradients
    a = np.abs(w).sum(axis=1)                # (R,C)
    s = w.sum(axis=1)
    xB = W + s                                # w_B
    cost = (a + np.abs(xB)) ** 2
    lam2 = 0.25 * float(cost.sum())
    sg_x = np.sign(xB)
    dlam2_dw = 0.5 * (a + np.abs(xB))[:, None, :] * (np.sign(w) + sg_x[:, None, :])
    dlam2_dW = 0.5 * (a + np.abs(xB)) * sg_x

    # h1' spectrum
    h1p = prob.H0 - np.einsum("rc,rcpq->pq", W, X, optimize=True)
    ev, Q = np.linalg.eigh(h1p)
    lam1 = float(np.abs(ev).sum())
    neg = float(-ev[ev < 0].sum())
    # smoothed sign / negative indicator to remove eigenvalue-crossing kicks
    soft = ev / np.sqrt(ev * ev + 0.05 * 0.05)
    M = (Q * soft[None, :]) @ Q.T                # smoothed polar matrix
    Pneg = (Q * (0.5 * (1.0 - soft))[None, :]) @ Q.T  # smoothed negative projector

    esos = prob.ecore - prob.eta * prob.beta_1 - 2.0 * neg - 0.5 * float((W * W).sum())
    egap = E_UPPER - esos
    Lam = lam1 + lam2
    leff_sq = egap * (2.0 * Lam - egap)           # smooth; may be negative out of domain

    out = dict(U=Un, X=X, XR=XR, G=G, D=D, e2=e2, e2sq=e2sq, lam2=lam2, lam1=lam1,
               Lam=Lam, esos=esos, egap=egap, leff_sq=leff_sq, h1p=h1p, M=M, Pneg=Pneg,
               a=a, s=s, xB=xB, cost=cost, dlam2_dw=dlam2_dw, dlam2_dW=dlam2_dW,
               w=w, W=W)
    return out


def objective(out: dict, t2: float, eg_lo: float, eg_hi: float, rho2: float,
              rhoE: float) -> dict:
    """Phi = Lambda1 + Lambda2 + rho2*ReLU(e2-t2)^2 + rhoE*(pen_lo^2 + pen_hi^2).

    W is projected onto sum W^2 = 1126.34 by the caller (then E_gap ~ 4.04 when
    all h1' eigenvalues are >= 0); the E_gap band penalty only corrects residual
    eigenvalue crossings, so the fit is decoupled from the E_gap oscillation.
    """
    pen2 = max(0.0, out["e2"] - t2)
    penE_lo = max(0.0, eg_lo - out["egap"])
    penE_hi = max(0.0, out["egap"] - eg_hi)
    phi = out["lam1"] + out["lam2"] + rho2 * pen2**2 + rhoE * (penE_lo**2 + penE_hi**2)
    dphi_egap = -2.0 * rhoE * penE_lo + 2.0 * rhoE * penE_hi

    W = out["W"]
    # gX = dLam1/dX + dphi_egap * dEg/dX + e2 barrier ;  dLam1/dX_rc = -W_rc M, dEg/dX = 2 W Pneg
    Dx = (out["D"] @ out["XR"].T).T.reshape(out["X"].shape)        # D @ x_rc
    gX = (-W[:, :, None, None] * out["M"]) \
        + dphi_egap * (2.0 * W[:, :, None, None] * out["Pneg"]) \
        + (2.0 * rho2 * pen2 / max(out["e2"], 1e-12)) * (2.0 * Dx)

    U, w = out["U"], out["w"]
    gw = out["dlam2_dw"] + np.einsum("rcpq,rbp,rbq->rbc", gX, U, U)
    dEg_dW = W + 2.0 * np.einsum("rcpq,pq->rc", out["X"], out["Pneg"])
    dLam1_dW = -np.einsum("rcpq,pq->rc", out["X"], out["M"])
    gW = out["dlam2_dW"] + dphi_egap * dEg_dW + dLam1_dW
    gU = 2.0 * np.einsum("rcpq,rbq,rbc->rbp", gX, U, w)
    gU = gU - np.sum(gU * U, axis=2, keepdims=True) * U
    return dict(phi=phi, pen2=pen2, penE=penE_lo, penE_hi=penE_hi, gU=gU, gw=gw, gW=gW)


# ---------------------------------------------------------------------------
# initialization from the spectrum
# ---------------------------------------------------------------------------

def init_W_h1cancel(U: np.ndarray, w: np.ndarray, H0: np.ndarray) -> np.ndarray:
    """W_init: W = alpha * w_ls with w_ls fitting sum W X ~ H0; alpha is the LARGEST
    root of E_gap(alpha) = 4.04 (all h1' eigenvalues >= 0 at the root), so E_gap
    starts exactly inside the valid domain with no transient."""
    R, C = w.shape[0], w.shape[2]
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = X.reshape(R * C, N, N)
    L = R * C
    Gll = np.einsum("cpq,dpq->cd", X, X)
    rhs = np.einsum("cpq,pq->c", X, H0)
    wls = np.linalg.solve(Gll + 1e-6 * np.eye(L) * (np.trace(Gll) / L + 1.0), rhs)
    SX = np.einsum("c,cpq->pq", wls, X)      # sum wls * X = H0 projection
    # E_gap(alpha) = -559.132 + 2*sum_{lam<0}|lam(H0 - alpha*SX)| + 0.5*alpha^2*|wls|^2
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
        idx = np.where(egs <= 4.04)[0]
        if len(idx) == 0:
            alpha = alpha0
        else:
            k = idx[-1]
            lo, hi = float(grid[k]), float(grid[k + 1])
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if egap_of(mid) >= 4.04:
                    hi = mid
                else:
                    lo = mid
            alpha = 0.5 * (lo + hi)
    return (wls * alpha).reshape(R, C)


def init_from_spectrum(prob: Problem, R: int, B: int, C: int, seed: int = 0) -> tuple:
    """Init v4: per-group joint diagonalizer + pair-type vectors, least-squares w.

    Each group keeps B/2 joint-diagonalizer vectors (diagonal coverage) and B/2
    pair vectors (e_i +- e_j)/sqrt(2) for the group's most important off-diagonal
    pairs.  w_c is the least-squares fit of sqrt(mu_j) V_j onto the group's
    span{u_b u_b^T} (full projection, not just the diagonal).
    """
    rng = np.random.default_rng(seed)
    L = R * C
    k = min(L, prob.ktop)
    assert k == L, f"spectrum has {prob.ktop} columns, need {L} (rerun b3_spectrum with --top {L})"
    mu = prob.mu[:L]
    Vm = np.asarray(prob.Vtop[:, :L]).T.reshape(L, N, N)
    U = np.zeros((R, B, N))
    w = np.zeros((R, B, C))
    half = B // 2
    for r in range(R):
        js = list(range(r * C, (r + 1) * C))
        A = np.einsum("jpq,jpr->qr", mu[js, None, None] * Vm[js], Vm[js])
        _, Ub = np.linalg.eigh(A)
        Ub = Ub[:, ::-1]  # descending eigenvalues
        U[r, :half] = Ub.T[:half]  # unit rows, orthonormal set
        # pair vectors for the top B/4 off-diagonal pairs by importance
        imp = np.zeros((N, N))
        for j in js:
            V = Vm[j]
            imp += mu[j] * np.abs(V)
        imp = imp - np.diag(np.diag(imp))
        imp = np.tril(imp) + np.triu(imp, 1).T * 0
        iu = np.triu_indices(N, 1)
        order = np.argsort(imp[iu])[::-1][: half // 2]
        for m, o in enumerate(order):
            i, kk = iu[0][o], iu[1][o]
            ep = np.zeros(N); ep[i] = 1.0; ep[kk] = 1.0
            em = np.zeros(N); em[i] = 1.0; em[kk] = -1.0
            U[r, half + 2 * m] = ep / np.sqrt(2.0)
            U[r, half + 2 * m + 1] = em / np.sqrt(2.0)
        # least-squares fit of each slice onto span{u_b u_b^T}
        Gmat = (U[r] @ U[r].T) ** 2
        Gmat = Gmat + 1e-6 * np.eye(B) * (np.trace(Gmat) / B + 1.0)
        for c, j in enumerate(js):
            rhs = np.sqrt(mu[j]) * np.einsum("bp,pq,bq->b", U[r], Vm[j], U[r])
            w[r, :, c] = np.linalg.solve(Gmat, rhs)
    W = init_W_h1cancel(U, w, prob.H0)
    return U, w, W


def init_resume(prob: Problem, R: int, B: int, C: int, path: str) -> tuple:
    """Resume from a b3_work checkpoint npz (U, w, W)."""
    data = np.load(RESULTS / path, allow_pickle=False)
    U = np.array(data["U"], dtype=np.float64)
    w = np.array(data["w"], dtype=np.float64)
    W = np.array(data["W"], dtype=np.float64)
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    return U, w, W


def init_from_baseline(prob: Problem, R: int, B: int, C: int) -> tuple:
    """Downscale the baseline: keep first R groups, first B columns, first C slices."""
    fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                   allow_pickle=False)
    U = np.array(fact["U"], dtype=np.float64)[:R, :B]
    w = np.array(fact["W"], dtype=np.float64)[:R, :B, :C]
    Wfull = np.array(fact["H2_I"], dtype=np.float64) - np.sum(np.array(fact["W"], dtype=np.float64), axis=1)
    W = Wfull[:R, :C]
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    return U, w, W


def init_baseline30(prob: Problem, R: int, B: int, C: int, seed: int = 0) -> tuple:
    """(65, B, 30) init: per-group rank-30 truncation of the baseline group Gram.

    Each group's 60 slices are re-factored as the top-30 SVD factors of the
    group's slice matrix, re-expressed in the group's own span{u_b u_b^T}
    (least squares).  Keeps ~all the recoverable mass instead of dropping 30
    arbitrary copies.  Then W = LS H0-cancel at E_gap ~ 4.04.
    """
    fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                   allow_pickle=False)
    U0 = np.array(fact["U"], dtype=np.float64)[:R, :B]
    U0 = U0 / np.linalg.norm(U0, axis=2, keepdims=True)
    w0 = np.array(fact["W"], dtype=np.float64)[:R, :B]
    C0 = w0.shape[2]
    U = np.zeros((R, B, N))
    w = np.zeros((R, B, C))
    for r in range(R):
        Xr = np.einsum("bp,bq,bc->cpq", U0[r], U0[r], w0[r], optimize=True)  # (C0,N,N)
        Xr = np.tril(Xr) + np.swapaxes(np.tril(Xr, k=-1), -1, -2)
        M = Xr.reshape(C0, N * N)
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)      # top-C0 factors
        U[r] = U0[r]
        Gmat = (U[r] @ U[r].T) ** 2
        Gmat = Gmat + 1e-6 * np.eye(B) * (np.trace(Gmat) / B + 1.0)
        for c in range(C):
            Xc = sv[c] * Vt[c].reshape(N, N)
            rhs = np.einsum("bp,pq,bq->b", U[r], Xc, U[r])
            w[r, :, c] = np.linalg.solve(Gmat, rhs)
    W = init_W_h1cancel(U, w, prob.H0)
    return U, w, W


def init_sector(prob: Problem, R: int, B: int, C: int, seed: int = 0) -> tuple:
    """Group 0 = exact Coulomb/diagonal sector (u = e_b); the rest fits the residual.

    Requires B == 76 so group 0 can hold the 76 canonical directions; the
    remaining R-1 groups use the v4 joint-diag + pair-vector scheme on the
    residual's top eigendirections.  Starts from e2 ~ ||residual|| (~1.9).
    """
    assert B == 76, "sector init requires B == 76"
    L = R * C
    U = np.zeros((R, B, N))
    w = np.zeros((R, B, C))
    # group 0: exact diagonal sector
    # C[p,s] = T[p,p,s,s] = tei[p,p,s,s] + 0.5*(S[p,p]+S[s,s])
    diag_pairs = np.array([p * N + p for p in range(N)])
    Cm = np.asarray(prob.T)[np.ix_(diag_pairs, diag_pairs)]
    evC, qC = np.linalg.eigh(Cm)
    U[0] = np.eye(N)
    for c in range(min(C, N)):
        w[0, :, c] = np.sqrt(float(np.maximum(evC[c], 0.0))) * qC[:, c]
    # residual D = T - G_0
    X0 = np.einsum("bp,bq,bc->cpq", U[0], U[0], w[0], optimize=True)
    X0 = np.tril(X0) + np.swapaxes(np.tril(X0, k=-1), -1, -2)
    XR0 = X0.reshape(-1, N * N)
    D = prob.T - XR0.T @ XR0
    print(f"  [sector init] residual norm = {np.sqrt(float((D*D).sum())):.6f}")
    muD, VD = np.linalg.eigh(D)
    order = np.argsort(muD)[::-1]
    muD = muD[order]
    VD = VD[:, order]
    k = L - C
    muD = muD[:k]
    Vm = VD[:, :k].T.reshape(k, N, N)
    half = B // 2
    for r in range(1, R):
        js = list(range((r - 1) * C, r * C))
        A = np.einsum("jpq,jpr->qr", muD[js, None, None] * Vm[js], Vm[js])
        _, Ub = np.linalg.eigh(A)
        Ub = Ub[:, ::-1]
        U[r, :half] = Ub.T[:half]
        imp = np.zeros((N, N))
        for j in js:
            imp += muD[j] * np.abs(Vm[j])
        imp = imp - np.diag(np.diag(imp))
        iu = np.triu_indices(N, 1)
        order = np.argsort(imp[iu])[::-1][: half // 2]
        for m, o in enumerate(order):
            i, kk = iu[0][o], iu[1][o]
            ep = np.zeros(N); ep[i] = 1.0; ep[kk] = 1.0
            em = np.zeros(N); em[i] = 1.0; em[kk] = -1.0
            U[r, half + 2 * m] = ep / np.sqrt(2.0)
            U[r, half + 2 * m + 1] = em / np.sqrt(2.0)
        Gmat = (U[r] @ U[r].T) ** 2
        Gmat = Gmat + 1e-6 * np.eye(B) * (np.trace(Gmat) / B + 1.0)
        for c, j in enumerate(js):
            rhs = np.sqrt(max(muD[j], 0.0)) * np.einsum("bp,pq,bq->b", U[r], Vm[j], U[r])
            w[r, :, c] = np.linalg.solve(Gmat, rhs)
    W = init_W_h1cancel(U, w, prob.H0)
    return U, w, W


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.ms: dict[int, np.ndarray] = {}
        self.vs: dict[int, np.ndarray] = {}

    def step(self, key: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        m = self.ms.get(key)
        if m is None:
            m = np.zeros_like(p)
            self.ms[key] = m
            self.vs[key] = np.zeros_like(p)
        v = self.vs[key]
        self.t += 1
        np.multiply(self.b1, m, out=m)
        np.add(m, (1 - self.b1) * g, out=m)
        np.multiply(self.b2, v, out=v)
        np.add(v, (1 - self.b2) * g * g, out=v)
        mh = m / (1 - self.b1**self.t)
        vh = v / (1 - self.b2**self.t)
        step = self.lr * mh / (np.sqrt(vh) + self.eps)
        return p - step


def report_metrics(prob: Problem, U: np.ndarray, w: np.ndarray, W: np.ndarray) -> dict:
    """Exact verifier metrics for the continuous state (error_1 = 0 by exact evecs)."""
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = np.einsum("tjp,tjq,tjl->tlpq", U, U, w, optimize=True)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_prime = dm.build_h1_prime(prob.h1_thc, prob.S, prob.beta_1,
                                 prob.ref_data["electrons"], W, X)
    _, evecs = np.linalg.eigh(h1_prime)
    u_all = np.vstack([evecs.T, U.reshape(-1, U.shape[-1])])
    return dm.evaluate_dfthc_vectors(
        u_all, w, W, prob.S, prob.beta_1, prob.ref_data, energy_upper=E_UPPER)


def grads_fit(out: dict, lam_reg: float, eg_lo: float, eg_hi: float, rhoE: float) -> dict:
    """Phase A: Phi = e2^2 + lam_reg*Lambda2 + rhoE*(pen_lo^2 + pen_hi^2)."""
    penE_lo = max(0.0, eg_lo - out["egap"])
    penE_hi = max(0.0, out["egap"] - eg_hi)
    phi = out["e2sq"] + lam_reg * out["lam2"] + rhoE * (penE_lo**2 + penE_hi**2)
    dphi_egap = -2.0 * rhoE * penE_lo + 2.0 * rhoE * penE_hi
    Dx = (out["D"] @ out["XR"].T).T.reshape(out["X"].shape)   # D @ x_rc
    # E_gap corrections flow through W only (no X coupling): the fit stays clean
    gX = 4.0 * Dx
    U, w = out["U"], out["w"]
    gw = lam_reg * out["dlam2_dw"] + np.einsum("rcpq,rbp,rbq->rbc", gX, U, U)
    dEg_dW = out["W"] + 2.0 * np.einsum("rcpq,pq->rc", out["X"], out["Pneg"])
    gW = lam_reg * out["dlam2_dW"] + dphi_egap * dEg_dW
    gU = 2.0 * np.einsum("rcpq,rbq,rbc->rbp", gX, U, w)
    gU = gU - np.sum(gU * U, axis=2, keepdims=True) * U
    return dict(phi=phi, penE=penE_lo, penE_hi=penE_hi, gU=gU, gw=gw, gW=gW)


def track_score(out: dict, t2: float, eg_lo: float) -> float:
    """Fixed-weight criterion for best-state tracking: leff^2 + barrier penalties."""
    return (out["leff_sq"] + 1e6 * max(0.0, out["e2"] - t2) ** 2
            + 1e4 * max(0.0, eg_lo - out["egap"]) ** 2
            + 1e4 * max(0.0, out["egap"] - 4.5) ** 2)


def run_opt(prob: Problem, R: int, B: int, C: int, steps_a: int, steps_b: int, lr: float,
            t2: float, eg_lo: float, rho2: float, rhoE: float,
            lam_reg: float = 0.05, init: str = "spectrum", out_name: str | None = None,
            log_every: int = 25, rho2_cap: float = 2e8, rho2_mult: float = 1.4,
            rho2_every: int = 40, freeze_u: bool = False, eg_hi: float = 15.0,
            resume_file: str | None = None) -> tuple:
    if init == "baseline":
        U, w, W = init_from_baseline(prob, R, B, C)
    elif init == "baseline30":
        U, w, W = init_baseline30(prob, R, B, C)
    elif init == "sector":
        U, w, W = init_sector(prob, R, B, C)
    elif init == "resume":
        U, w, W = init_resume(prob, R, B, C, resume_file)
    else:
        U, w, W = init_from_spectrum(prob, R, B, C)
    print(f"[opt] init '{init}' (R,B,C)=({R},{B},{C}) stepsA={steps_a} stepsB={steps_b} "
          f"lr={lr} t2={t2} eg_lo={eg_lo} eg_hi={eg_hi} lam_reg={lam_reg} freeze_u={freeze_u}")
    out = forward(prob, U, w, W)
    print(f"  init: e2={out['e2']:.6f} Lam={out['Lam']:.4f} Eg={out['egap']:.4f} "
          f"leff_sq={out['leff_sq']:.1f}")

    adam = Adam(lr=lr)
    best = (float("inf"), None)
    t_last = time.time()
    wsphere = 1126.34  # sum W^2 keeping E_gap ~ 4.04 when h1' eigenvalues >= 0

    def step_common(U_, w_, W_):
        if not freeze_u:
            U_ = adam.step(0, U_, res["gU"])
            U_ = U_ / np.linalg.norm(U_, axis=2, keepdims=True)
        w_ = adam.step(1, w_, res["gw"])
        sw2 = float(np.sum(W_ * W_))
        gWt = res["gW"] - np.sum(res["gW"] * W_) / max(sw2, 1e-12) * W_  # tangent
        W_ = adam.step(2, W_, gWt)
        W_ = W_ * np.sqrt(wsphere / max(float(np.sum(W_ * W_)), 1e-12))
        return U_, w_, W_

    # ---- phase A: pure fit ----
    for it in range(steps_a):
        out = forward(prob, U, w, W)
        res = grads_fit(out, lam_reg, eg_lo, eg_hi, rhoE)
        U, w, W = step_common(U, w, W)
        sc = track_score(out, t2, eg_lo)
        if sc < best[0]:
            best = (sc, (U.copy(), w.copy(), W.copy()))
        if out_name and it > 0 and it % 200 == 0:
            RESULTS.mkdir(exist_ok=True)
            np.savez(RESULTS / f"{out_name}_ckpt.npz", U=U, w=w, W=W)
        if it % log_every == 0 or it == steps_a - 1:
            dt = time.time() - t_last
            t_last = time.time()
            print(f"  A {it:4d} e2={out['e2']:.6f} Lam2={out['lam2']:9.2f} Eg={out['egap']:8.3f} "
                  f"track={sc:12.1f} ({dt:.1f}s)")
    # ---- phase B: lambda_eff tune with error barrier ----
    for it in range(steps_b):
        out = forward(prob, U, w, W)
        res = objective(out, t2, eg_lo, eg_hi, rho2, rhoE)
        U, w, W = step_common(U, w, W)
        sc = track_score(out, t2, eg_lo)
        if sc < best[0]:
            best = (sc, (U.copy(), w.copy(), W.copy()))
        if rho2_every and it % rho2_every == rho2_every - 1 and rho2 < rho2_cap:
            rho2 = min(rho2 * rho2_mult, rho2_cap)
        if it % 80 == 79 and rhoE < 2e5:
            rhoE = min(rhoE * 1.3, 2e5)
        if it % log_every == 0 or it == steps_b - 1:
            dt = time.time() - t_last
            t_last = time.time()
            print(f"  B {it:4d} e2={out['e2']:.6f} Lam={out['Lam']:9.3f} Eg={out['egap']:7.4f} "
                  f"leff2={out['leff_sq']:10.2f} track={sc:12.1f} rho2={rho2:.1e} ({dt:.1f}s)")
        if out_name and it > 0 and it % 200 == 0:
            RESULTS.mkdir(exist_ok=True)
            np.savez(RESULTS / f"{out_name}_ckpt.npz", U=U, w=w, W=W)
    if best[1] is None:  # zero-step edge case: report the init state
        U, w, W = U / np.linalg.norm(U, axis=2, keepdims=True), w, W
        best = (float("inf"), (U.copy(), w.copy(), W.copy()))
    Ub, wb, Wb = best[1]
    if out_name:
        RESULTS.mkdir(exist_ok=True)
        np.savez(RESULTS / f"{out_name}.npz", U=Ub, w=wb, W=Wb)
        print(f"[opt] saved best to results/{out_name}.npz")
    m = report_metrics(prob, Ub, wb, Wb)
    print(f"[opt] best(verifier): e2={m['error_2']:.6f} e1={m['error_1']:.2e} "
          f"Lam={m['Lambda']:.4f} E_gap={m['E_gap']:.4f} leff={m['lambda_eff']:.4f}")
    return Ub, wb, Wb


# ---------------------------------------------------------------------------
# gradient check
# ---------------------------------------------------------------------------

def check_grads(prob: Problem) -> None:
    R, B, C = 2, 3, 2
    rng = np.random.default_rng(0)
    U = rng.standard_normal((R, B, N))
    U /= np.linalg.norm(U, axis=2, keepdims=True)
    w = rng.standard_normal((R, B, C)) * 0.3
    W = rng.standard_normal((R, C)) * 0.5
    t2, eg_lo, eg_hi, rho2, rhoE = 0.016, 3.93, 20.0, 1e3, 1e2

    def phi_of(U_, w_, W_):
        out = forward(prob, U_, w_, W_)
        return objective(out, t2, eg_lo, eg_hi, rho2, rhoE)["phi"]

    out = forward(prob, U, w, W)
    res = objective(out, t2, eg_lo, eg_hi, rho2, rhoE)
    print(f"phi = {res['phi']:.8f}, e2={out['e2']:.6f}, egap={out['egap']:.4f}, Lam={out['Lam']:.4f}")
    eps = 1e-6
    for name, g, p, key in [("U", res["gU"], U, 0), ("w", res["gw"], w, 1), ("W", res["gW"], W, 2)]:
        flat_p = p.ravel()
        flat_g = g.ravel()
        i = rng.integers(0, flat_p.size, 24)
        errs = []
        for idx in i:
            pp = flat_p.copy()
            pp[idx] += eps
            pm = flat_p.copy()
            pm[idx] -= eps
            if key == 0:
                Up, Um = pp.reshape(p.shape), pm.reshape(p.shape)
                Up /= np.linalg.norm(Up, axis=2, keepdims=True)
                Um /= np.linalg.norm(Um, axis=2, keepdims=True)
                fd = (phi_of(Up, w, W) - phi_of(Um, w, W)) / (2 * eps)
            elif key == 1:
                fd = (phi_of(U, pp.reshape(p.shape), W) - phi_of(U, pm.reshape(p.shape), W)) / (2 * eps)
            else:
                fd = (phi_of(U, w, pp.reshape(p.shape)) - phi_of(U, w, pm.reshape(p.shape))) / (2 * eps)
            errs.append(abs(fd - flat_g[idx]) / max(1e-9, abs(fd), abs(flat_g[idx])))
        print(f"  {name}: median rel err = {np.median(errs):.2e}, max = {max(errs):.2e}")
        assert np.median(errs) < 1e-4, f"gradient check failed for {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, nargs=3, default=[16, 24, 24])
    ap.add_argument("--steps-a", type=int, default=600, help="phase A (fit) steps")
    ap.add_argument("--steps-b", type=int, default=800, help="phase B (lambda_eff) steps")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--t2", type=float, default=0.0165, help="error_2 soft target")
    ap.add_argument("--eg-lo", type=float, default=3.925, help="E_gap lower target")
    ap.add_argument("--eg-hi", type=float, default=4.5, help="E_gap upper band edge")
    ap.add_argument("--rho2", type=float, default=1e6)
    ap.add_argument("--rhoE", type=float, default=2e4)
    ap.add_argument("--lam-reg", type=float, default=0.05, help="Lambda2 weight in phase A")
    ap.add_argument("--freeze-u", action="store_true", help="do not update U")
    ap.add_argument("--init", choices=["spectrum", "baseline", "baseline30", "sector", "resume"], default="spectrum")
    ap.add_argument("--resume-file", default=None, help="b3_work npz for --init resume")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-grads", action="store_true")
    ap.add_argument("--eval-only", default=None, help="path of results/b3_work_*.npz to re-report")
    args = ap.parse_args()

    prob = Problem()
    if args.check_grads:
        check_grads(prob)
        return
    R, B, C = args.config
    if args.eval_only:
        data = np.load(RESULTS / args.eval_only, allow_pickle=False)
        m = report_metrics(prob, data["U"], data["w"], data["W"])
        print(f"e2={m['error_2']:.6f} e1={m['error_1']:.2e} Lam={m['Lambda']:.4f} "
              f"E_gap={m['E_gap']:.4f} leff={m['lambda_eff']:.4f}")
        return
    out_name = args.out or f"b3_work_{R}_{B}_{C}"
    run_opt(prob, R, B, C, args.steps_a, args.steps_b, args.lr, args.t2, args.eg_lo,
            args.rho2, args.rhoE, lam_reg=args.lam_reg, init=args.init, out_name=out_name,
            freeze_u=args.freeze_u, eg_hi=args.eg_hi, resume_file=args.resume_file)


if __name__ == "__main__":
    main()
