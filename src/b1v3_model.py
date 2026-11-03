# -*- coding: utf-8 -*-
"""Benchmark I v3: centered signed-digit quadratic Horner, exact integer model.

Independent reimplementation of the signed-digit architecture for 1/sqrt(x)
(a technique first used publicly for this benchmark in the "VQC Verified
Optimization Atlas" dataset, Apache-2.0; see data/community/toolbox). The
mathematics and the coefficient search below are re-derived and re-coded here.

Data flow
---------
For x in [1, 3,133,452], width 22:
  s    = 21 - floor(log2 x)                    (normalization shift, 0..21)
  xn   = x << s,  so xn in [2^21, 2^22)        (top bit 1)
  e    = s & 1                                 (parity of the shift)
  a    = (xn >> 14) & 127                      (segment index, 128 segments)
  d    = xn & 0x3FFF                           (14-bit local coordinate)
  1/sqrt(x) = 2^(s/2)/sqrt(xn) = 2^(s/2) 2^-10.5 / sqrt(M),  M = xn 2^-21.

Each segment (e, a) stores an integer quadratic for y0(M) = 2^((e-1)/2)/sqrt(M),
evaluated in *centered signed digits*:

  sig_i = 2 bit_i(d) - 1            (i = 0..13),  so that
  sum_i sig_i 2^(i-21) = 2 (d 2^-21 - 8191.5 2^-21) = 2 (delta - center)

with the center 8191.5 2^-21 = (2^14-1)/2^22 absorbed into the fitted
coefficients.  The evaluation

  R = K1 - sum_i sig_i round(K2 / 2^sh2_i)          (rows: sh2_i < F2-1, minus drop)
  Q = sum_i sig_i round(R  / 2^sh1_i)               (all 14 rows, low to high)
  Y = (K0 - Q) >> GY

gives Y / 2^FYR ~ scale / sqrt(M).  Each row is one conditional add/subtract of
a shifted copy -- no general multiplier -- implemented in the circuit by a
Gidney carry-ripple adder uncomputed by measurement.

Phasing:  phase = Y 2^((s-e)/2) 2^(GY-FYR-10)  (turns), applied as
  units = (Y << (27 + s//2)) mod 2^64   for FYR=27, GY=1,
checked against the official reference values (isqrt formula).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

W = 22          # x register width
DB = 14         # local-coordinate bits
MB = 21         # M = xn * 2^-MB
NSEG = 128      # segments per shift parity
MAX_X = 3 * (2**10 - 2) ** 2
MASK64 = (1 << 64) - 1


@dataclasses.dataclass(frozen=True)
class V3Cfg:
    f2: int = 8         # K2 integer bits
    f1: int = 19        # K1 / R scale
    fyr: int = 27       # Y fractional scale
    gy: int = 1         # low bits dropped of y0 before phasing
    drop2: tuple = (5,) # R rows omitted
    lb: int = 21        # low-word register width
    s: int = 20         # split between high/low words

    @property
    def fy(self) -> int:
        return self.fyr + self.gy

    @property
    def w2(self) -> int:
        return self.f2 - 1

    def sh2(self, i: int) -> int:
        return self.f2 + W - i - self.f1

    def sh1(self, i: int) -> int:
        return self.f1 + W - i - self.fy

    def rrows(self) -> list:
        return [i for i in reversed(range(DB)) if i not in self.drop2 and self.sh2(i) < self.w2]


def rd(v, s):
    """round(v / 2^s) for integer v; negative s shifts left."""
    if s < 0:
        return v << (-s)
    r = v >> s
    if s >= 1:
        r = r + ((v >> (s - 1)) & 1)
    return r


def reference_values(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """target mod 1 and tol(x) in units of 2^-64 turns (official formula)."""
    rs = [math.isqrt((1 << 130) // int(x)) for x in xs.tolist()]
    target = np.array([((r + 1) >> 1) & MASK64 for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> 25, 1 << 63) for r in rs], dtype=np.uint64)
    return target, tol


# ------------------------------------------------------------------ fitting

def lawson_fit(lo: float, hi: float, mid: float, scale: float, deg: int = 2,
               iters: int = 80, n: int = 3001) -> np.ndarray:
    """Weighted (Lawson) minimax quadratic fit of scale/sqrt(M) on [lo, hi]."""
    f = lambda m: scale / np.sqrt(m)
    d = np.linspace(lo - mid, hi - mid, n)
    A = np.vander(d, deg + 1, increasing=True)
    w = np.sqrt(mid + d) / scale          # relative-error weight
    wt = np.ones_like(d)
    for _ in range(iters):
        sw = w * np.sqrt(wt)
        coef = np.linalg.lstsq(A * sw[:, None], f(mid + d) * sw, rcond=None)[0]
        err = np.abs((A @ coef - f(mid + d)) * w)
        wt = wt * err
        wt /= wt.sum()
    return coef


def pipeline(cfg: V3Cfg, k2: int, k1: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R and Q for all grid values d in [0, 2^DB), exact integer arithmetic."""
    d = np.asarray(d, dtype=np.int64)
    sig = 2 * ((d[:, None] >> np.arange(DB, dtype=np.int64)) & 1) - 1
    r = np.full_like(d, k1, dtype=np.int64)
    for i in cfg.rrows():
        r = r - sig[:, i] * rd(k2, cfg.sh2(i))
    q = np.zeros_like(d, dtype=np.int64)
    for i in reversed(range(DB)):
        q = q + sig[:, i] * rd(r, cfg.sh1(i))
    return r, q


def reachable_d(e: int, a: int) -> np.ndarray:
    """Grid values of d reachable for segment (e, a) over the promised domain.

    x = xn >> s with xn = x << s in [2^21, 2^22) and s = 22 - bit_length(x):
      - odd s (e=1): s = 1 already covers every even d;
      - even s (e=0): s = 0 requires x = xn <= MAX_X, so segments with
        a >= (MAX_X - 2^21) >> 14 = 63 are only reachable through s >= 2,
        i.e. d divisible by 4 (a = 63 keeps the low d <= MAX_X - 2^21 - 63*2^14
        as well).
    """
    if e == 1:
        return np.arange(0, 2**DB, 2, dtype=np.int64)
    if a < 63:
        return np.arange(2**DB, dtype=np.int64)
    d4 = np.arange(0, 2**DB, 4, dtype=np.int64)
    if a == 63:
        d_hi = MAX_X - (1 << 21) - 63 * (1 << 14)
        return np.concatenate([d4, np.arange(d_hi + 1, dtype=np.int64)])
    return d4


def fit_segment(cfg: V3Cfg, e: int, a: int, search: int = 4, k2span: int = 2) -> tuple[int, int, int, float]:
    """Integer (K2, K1, K0) minimizing the worst relative phase error over the grid."""
    scale = 2.0 ** ((e - 1) / 2)
    h = 2.0 ** (DB - MB)                       # 2^-7
    lo = 1.0 + a * h
    mid = lo + (2**DB - 1) / 2.0 ** (MB + 1)
    c = lawson_fit(lo, lo + h, mid, scale)
    c1, c2 = -c[1], c[2]
    k2c = int(round(c2 * 2**cfg.f2))
    k1c = int(round(c1 * 2**cfg.f1))

    d = reachable_d(e, a)
    m = lo + d * 2.0**-MB
    tgt = scale / np.sqrt(m) * 2.0**cfg.fy
    wgt = np.sqrt(m) / scale                 # = 2^fy / tgt

    best = None
    for k2 in range(k2c - k2span, k2c + k2span + 1):
        for k1 in range(k1c - search, k1c + search + 1):
            r, q = pipeline(cfg, k2, k1, d)
            if (r < 0).any() or r.max() >= (1 << 22):
                continue
            t = tgt + q
            a_, b_ = math.floor(float(t.min())), math.ceil(float(t.max()))

            def mx(k0):
                y = ((k0 - q) >> cfg.gy) << cfg.gy
                return float(np.max(np.abs(y - tgt) * wgt))

            while b_ - a_ > 2:               # ternary search on unimodal K0
                m1 = a_ + (b_ - a_) // 3
                m2 = b_ - (b_ - a_) // 3
                if mx(m1) < mx(m2):
                    b_ = m2
                else:
                    a_ = m1
            for k0 in range(a_, b_ + 1):
                v = mx(k0)
                if best is None or v < best[0]:
                    best = (v, k2, k1, k0)
    return best  # (metric, k2, k1, k0)


def fit_all(cfg: V3Cfg, search: int = 4, verbose: bool = True) -> dict:
    tabs = {}
    worst = 0.0
    for e in (0, 1):
        for a in range(NSEG):
            v, k2, k1, k0 = fit_segment(cfg, e, a, search)
            tabs[(e, a)] = (k2, k1, k0)
            worst = max(worst, v)
            if verbose and a % 32 == 0:
                print(f"  e={e} a={a:3d}: metric {v/2**cfg.fy:.3e}")
    return tabs, worst


# ------------------------------------------------------------------- layout

def make_layout(cfg: V3Cfg, tabs: dict, verbose: bool = True) -> dict:
    """Register widths and the (low, high) split of K0 for the circuit."""
    rmax = 0
    per = {}
    for (e, a), (k2, k1, k0) in tabs.items():
        d = reachable_d(e, a)
        r, q = pipeline(cfg, k2, k1, d)
        if (r < 0).any():
            raise ValueError(f"negative R at {(e, a)}")
        rmax = max(rmax, int(r.max()), k1)
        per[(e, a)] = (int(q.min()), int(q.max()))
    w1 = rmax.bit_length()
    lo_tab, hi_tab = {}, {}
    for key, (k2, k1, k0) in tabs.items():
        qmin, qmax = per[key]
        base = k0 % (1 << cfg.s)
        j = 0
        while base + j * (1 << cfg.s) < qmax:
            j += 1
        k0lo = base + j * (1 << cfg.s)
        # Ylow = k0lo - Q must fit the LB-bit low register.  Note that Ylow may
        # exceed 2^S: bit S of the low word and bit 0 of the high word are then
        # both phased at weight 2^(S-38+shift), and their sum is exactly the
        # carry into bit S of K0 - Q (intentional overlap-add at the seam).
        if k0lo - qmin >= (1 << cfg.lb):
            raise ValueError(f"layout infeasible at {key}: k0lo-qmin={k0lo-qmin}")
        lo_tab[key] = k0lo
        hi_tab[key] = (k0 - k0lo) >> cfg.s
        if hi_tab[key] < 0:
            raise ValueError(f"negative hi at {key}")
    whi = max(max(hi_tab.values()).bit_length(), 1)
    if verbose:
        print(f"layout: w1={w1} whi={whi} LB={cfg.lb} S={cfg.s} rmax={rmax}")
    return dict(w1=w1, whi=whi, lo=lo_tab, hi=hi_tab)


# ----------------------------------------------------------- full-domain check

def full_domain_check(cfg: V3Cfg, tabs: dict, chunk: int = 1 << 16) -> dict:
    """Exact phase check over every x; returns worst ratio and min margin."""
    table = np.array([tabs[(e, a)] for e in (0, 1) for a in range(NSEG)], dtype=np.int64)
    worst = 0.0
    wx = 0
    margin = 1 << 64
    for lo in range(1, MAX_X + 1, chunk):
        x = np.arange(lo, min(lo + chunk, MAX_X + 1), dtype=np.int64)
        s = 21 - np.floor(np.log2(x)).astype(np.int64)
        xn = x << s
        e = s & 1
        a = (xn >> 14) & 127
        d = xn & 16383
        k2, k1, k0 = table[128 * e + a].T
        sig = 2 * ((d[:, None] >> np.arange(DB)) & 1) - 1
        r = np.full_like(d, k1, dtype=np.int64)
        for i in cfg.rrows():
            r = r - sig[:, i] * rd(k2, cfg.sh2(i))
        q = np.zeros_like(d, dtype=np.int64)
        for i in reversed(range(DB)):
            q = q + sig[:, i] * rd(r, cfg.sh1(i))
        y = (k0 - q) >> cfg.gy
        if (y <= 0).any() or (y >= (1 << cfg.fyr)).any():
            j = int(np.flatnonzero((y <= 0) | (y >= (1 << cfg.fyr)))[0])
            raise ValueError(f"y out of range at x={x[j]}: {y[j]}")
        units = (y.astype(np.uint64) << np.uint64(27 + s // 2)) & np.uint64(MASK64)
        target, tol = reference_values(x.astype(np.uint64))
        used = np.abs((units - target).view(np.int64)).view(np.uint64)
        if (used >= tol).any():
            j = int(np.flatnonzero(used >= tol)[0])
            raise ValueError(f"phase fail at x={x[j]}: ratio {used[j]/tol[j]:.3f}")
        ratio = used / tol
        j = int(ratio.argmax())
        if ratio[j] > worst:
            worst = float(ratio[j])
            wx = int(x[j])
        margin = min(margin, int((tol - used).min()))
    n_rot = None
    return dict(worst=worst, worst_x=wx, margin_units=margin)


# --------------------------------------------------------------- cost model

def cost_estimate(cfg: V3Cfg, lay: dict, n_rot: int = 45) -> dict:
    """Static Toffoli estimate (calibrated against measured builds)."""
    w1, whi = lay["w1"], lay["whi"]
    n_rrows = len(cfg.rrows())
    # per-row adder cost ~ w1-1 for R rows, LB-1 for Y rows (folded Gidney)
    rrows = 2 * n_rrows * (w1 - 1)
    yrows = 2 * DB * (cfg.lb - 1)
    shifts = 2 * (sum(min(20, 1) + 0 for _ in ()) or 0)  # placeholder
    # measured components (see builder marks)
    qrom = 254 + 2 * 95
    norm = 210
    est = rrows + yrows + qrom + norm
    return dict(rrows=rrows, yrows=yrows, qrom=qrom, norm=norm, static=est)


if __name__ == "__main__":
    cfg = V3Cfg()
    print("fitting", NSEG * 2, "segments ...")
    tabs, worst_metric = fit_all(cfg, verbose=False)
    print(f"worst fit metric: {worst_metric / 2**cfg.fy:.4e} "
          f"({worst_metric / 2**cfg.fy / 2**-24:.3f} x 2^-24)")
    lay = make_layout(cfg, tabs)
    print("full domain check ...")
    res = full_domain_check(cfg, tabs)
    print(f"worst error/tol = {res['worst']:.4f} at x={res['worst_x']}")
    print(f"min margin = {res['margin_units']} units (2^-64 turns)")
