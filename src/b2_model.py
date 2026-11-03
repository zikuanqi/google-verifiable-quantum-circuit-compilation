"""b2_model.py — Classical register-level model of the reversible Berlekamp-Massey
Reed-Solomon MBUC decoder for Benchmark II (DQI), per Gu et al. arXiv:2610.22367.

Stages (all simulated value-for-value, forward AND reverse, exactly as the
circuit will execute them):

  1. Forward BM (70 rounds, rescaled updates, shared MADD + conditional
     routing, delta recovered from sigma0, 1-entry discrepancy history stack
     with XOR-push; overflow and cutoff violations -> declared failure).
  2. Reverse BM (branch detection via tau[0] != 0).
  3. In-place omega product over all 70 syndrome words (S36..S70 -> 0).
  4. sqrt permutation + linear maps: sigma -> (E, O) stored at sigma[0..17],
     sigma[18..35].
  5. Aligned fixed-schedule EEA on (E, O): 35 fixed iterations, quotients
     parked in the words they clear, boundary swaps gated on a rotation
     counter c, done-flag on zero remainder; then a re-run that recomputes
     the quotients and builds A, B via the continued fraction into the
     zeroed syndrome words (A: 18 words, B: 17 words).
  6. Chien scan over all 4095 positions: root test sigma(beta)==0 and phase
     bit Tr(mu_j * omega(beta) * psi(beta)) (the circuit applies this via a
     rank-1 bilinear decomposition with 12 CCZ gates).
  7. Exact mirrors of stages 3-5 and the full restoration check.

All arithmetic is GF(2^12) with the verifier's field: modulus 0x1053,
primitive element 2, little-endian polynomial basis.
"""

from __future__ import annotations

import time

import numpy as np

MODULUS = 0x1053
DEGREE = 12
ORDER = (1 << DEGREE) - 1
PRIMITIVE = 2


# ----------------------------------------------------------------------------
# GF(2^12)
# ----------------------------------------------------------------------------

class GF12:
    def __init__(self):
        self.antilog = [0] * ORDER
        v = 1
        for k in range(ORDER):
            self.antilog[k] = v
            v = _mul_raw(v, PRIMITIVE)
        assert len(set(self.antilog)) == ORDER
        self.log = [0] * (1 << DEGREE)
        for k, a in enumerate(self.antilog):
            self.log[a] = k
        self._dual = None

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.antilog[(self.log[a] + self.log[b]) % ORDER]

    def inv(self, a):
        if a == 0:
            return 0
        return self.antilog[(ORDER - self.log[a]) % ORDER]

    def pow(self, a, e):
        if a == 0:
            return 0
        return self.antilog[(self.log[a] * e) % ORDER]

    def sq(self, a):
        return self.mul(a, a)

    def sqrt(self, a):
        # inverse of Frobenius: x^(2^11) in GF(2^12)
        return self.pow(a, 1 << (DEGREE - 1))

    def dual_basis(self):
        if self._dual is not None:
            return self._dual
        basis = [1 << k for k in range(DEGREE)]
        tr = lambda a: a.bit_count() & 1
        B = np.zeros((DEGREE, DEGREE), dtype=np.uint8)
        for i in range(DEGREE):
            for j in range(DEGREE):
                B[i, j] = tr(_mul_raw(basis[i], basis[j]))
        Binv = _gf2_inv_matrix(B)
        dual = []
        for k in range(DEGREE):
            mu = 0
            for j in range(DEGREE):
                if Binv[j, k]:
                    mu ^= 1 << j
            dual.append(mu)
        self._dual = dual
        return dual

    def mask_to_mu(self, mask):
        mu = 0
        for k in range(DEGREE):
            if (mask >> k) & 1:
                mu ^= self.dual_basis()[k]
        return mu


def _mul_raw(a, b):
    p = 0
    while b:
        if b & 1:
            p ^= a
        b >>= 1
        a <<= 1
        if a & (1 << DEGREE):
            a ^= MODULUS
    return p


def _gf2_inv_matrix(B):
    n = B.shape[0]
    A = np.hstack([B.copy(), np.eye(n, dtype=np.uint8)])
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r, col]), None)
        assert piv is not None
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
        for r in range(n):
            if r != col and A[r, col]:
                A[r] ^= A[col]
    return A[:, n:]


# ----------------------------------------------------------------------------
# DQI input distribution (replicates verify_submission_dqi.sample_shots)
# ----------------------------------------------------------------------------

def optimal_dqi_weights(m, max_errors, q, r):
    k = np.arange(max_errors + 1, dtype=np.float64)
    off = np.sqrt(k[1:] * (m - k[1:] + 1))
    d = (q - 2 * r) / np.sqrt(r * (q - r))
    _, vecs = np.linalg.eigh(np.diag(k * d) + np.diag(off, 1) + np.diag(off, -1))
    w = np.abs(vecs[:, -1])
    return w / np.linalg.norm(w)


class DQIDistribution:
    def __init__(self, m=4095, n=70, degree=DEGREE, r=2016, gf=None):
        self.m, self.n, self.degree, self.r = m, n, degree, r
        self.gf = gf or GF12()
        w = optimal_dqi_weights(m, n // 2, 1 << degree, r)
        self.weight_probs = w ** 2
        self.mean_weight = float(np.arange(len(w)) @ self.weight_probs)

    def sample(self, rng):
        gf = self.gf
        m, n = self.m, self.n
        k = int(rng.choice(len(self.weight_probs), p=self.weight_probs))
        pos = rng.choice(m, size=k, replace=False)
        max_value = (1 << DEGREE) - 1
        vals = rng.integers(1, max_value, size=k, endpoint=True)
        mbuc = rng.integers(0, max_value, size=m, endpoint=True)
        syndromes = [0] * n
        for p, v in zip(pos, vals):
            lv = gf.log[v]
            for i in range(1, n + 1):
                syndromes[i - 1] ^= gf.antilog[(lv + i * int(p)) % ORDER]
        errors = {int(p): int(v) for p, v in zip(pos, vals)}
        phase = 0
        for p, v in errors.items():
            phase ^= (v & mbuc[p]).bit_count() & 1
        return dict(weight=k, errors=errors, syndromes=syndromes,
                    mbuc=[int(x) for x in mbuc], phase=bool(phase))


# ----------------------------------------------------------------------------
# Rescaled BM (register-level)
# ----------------------------------------------------------------------------

class BMState:
    def __init__(self, gf, cutoffs, n=70, caps=3):
        self.gf = gf
        self.n = n
        self.cutoffs = cutoffs
        self.caps = caps
        self.sigma = [0] * 36
        self.tau = [0] * 36
        self.sigma[0] = 1
        self.tau[0] = 1
        self.d = 0
        self.stack = []          # shift stack of (round, delta); overflow -> fail
        self.fail_stack = False
        self.fail_cutoff = False
        self.ev_shift_top = 0

    def round_fwd(self, r, syndromes):
        gf = self.gf
        Km = min(self.cutoffs[r - 1], 35)
        K = min(Km, r - 1)

        # shift tau (cyclic left rotate; old tau[35] lands in tau[0])
        old_t35 = self.tau[35]
        for i in range(35, 0, -1):
            self.tau[i] = self.tau[i - 1]
        self.tau[0] = old_t35
        if old_t35 != 0:
            self.ev_shift_top += 1

        # discrepancy
        delta = gf.mul(self.sigma[0], syndromes[r - 1])
        for i in range(1, K + 1):
            delta ^= gf.mul(self.sigma[i], syndromes[r - 1 - i])

        for i in range(Km + 1, 36):
            if self.sigma[i] or self.tau[i]:
                self.fail_cutoff = True

        nz = delta != 0
        rep = nz and self.d >= 0

        alpha = gf.mul(delta, self.sigma[0])
        alpha_inv = gf.inv(alpha)
        m = alpha_inv if rep else alpha

        if rep:
            for i in range(Km + 1):
                self.sigma[i], self.tau[i] = self.tau[i], self.sigma[i]

        for i in range(Km + 1):
            self.sigma[i] ^= gf.mul(m, self.tau[i])

        # fused-Q push on EVERY d < 0 round (reference scheme): shift the
        # stack down, insert (r, delta) at the top; overflow -> declared fail.
        if self.d < 0:
            self.stack.append((r, delta))
            if len(self.stack) > self.caps:
                self.fail_stack = True

        if rep:
            self.d = -self.d
        else:
            self.d += 1
        return delta, rep

    def round_rev(self, r, syndromes):
        gf = self.gf
        Km = min(self.cutoffs[r - 1], 35)

        # entry: d is the forward post-update value.  Undo the +1 to reach
        # the complemented value, whose sign is [rep OR pushed].
        self.d -= 1
        entry = self.stack[-1] if self.stack else None
        pushed = entry is not None and entry[0] == r
        rep = (self.d < 0) and not pushed
        rt = pushed

        if rep:
            m = gf.mul(self.sigma[0], gf.inv(self.tau[0]))
        else:
            sdelta = entry[1] if rt else 0
            m = gf.mul(sdelta, self.sigma[0])

        if rep or rt:
            for i in range(Km + 1):
                self.sigma[i] ^= gf.mul(m, self.tau[i])
            if rep:
                for i in range(Km + 1):
                    self.sigma[i], self.tau[i] = self.tau[i], self.sigma[i]
            if rt:
                self.stack.pop()

        old_t0 = self.tau[0]
        for i in range(35):
            self.tau[i] = self.tau[i + 1]
        self.tau[35] = old_t0

        # restore the pre-round d: undo the complement under rep
        if rep:
            self.d = -self.d - 1
        return m, rep, rt


# ----------------------------------------------------------------------------
# omega product
# ----------------------------------------------------------------------------

def omega_product(syndromes, sigma, K_cut, gf, *, reverse=False):
    n = len(syndromes)
    if reverse:
        # mirror: ascending order, reading already-restored originals
        for k in range(1, n):
            for j in range(1, min(k, K_cut) + 1):
                syndromes[k] ^= gf.mul(sigma[j], syndromes[k - j])
    else:
        # descending order so the in-place reads see the ORIGINAL syndromes
        for k in range(n - 1, 0, -1):
            for j in range(1, min(k, K_cut) + 1):
                syndromes[k] ^= gf.mul(sigma[j], syndromes[k - j])


# ----------------------------------------------------------------------------
# sqrt transform: sigma -> (E, O); E at sigma[0..17], O at sigma[18..35]
# (a fixed permutation plus linear maps; in the circuit: SWAPs + CNOTs)
# ----------------------------------------------------------------------------

def sigma_to_eo(sigma, gf):
    E = [gf.sqrt(sigma[2 * i]) for i in range(18)]
    O = [gf.sqrt(sigma[2 * i + 1]) for i in range(18)]
    return E, O


def eo_to_sigma(E, O, gf):
    sigma = [0] * 36
    for i in range(18):
        sigma[2 * i] = gf.sq(E[i])
        sigma[2 * i + 1] = gf.sq(O[i])
    return sigma


# ----------------------------------------------------------------------------
# Aligned fixed-schedule EEA on (E, O)
#
# r0 = E aligned at word 17 (leading coeff at word 17), r1 = O aligned.
# 35 fixed iterations; each:
#     q := r0[17]/r1[17]
#     r0 ^= q * r1                      (18 words)
#     park: r0[17] ^= q                 (word was just cleared)
#     rotate r0 left by 1               (cyclic)
#     c += 1
#     boundary: if (not done) and c >= d0-d1+1 and ((r0[17]!=0) or
#                (d1==0 and r0==0)):
#         swap r0<->r1; d0,d1 = d1, d0-c; c = 0; done = (r1==0)
#
# After the schedule: r0 = gcd (aligned), r1 = 0, quotients parked inside the
# arrays.  Then the reconstruction re-runs the same 35 iterations recomputing
# q and applying the continued fraction (A,B) <- (q*z*A + B, A) gated on q!=0,
# building A (18 words) and B (17 words) right-aligned.
# ----------------------------------------------------------------------------

def _poly_zero(p):
    return all(x == 0 for x in p)


def _poly_deg(p):
    for i in range(len(p) - 1, -1, -1):
        if p[i]:
            return i
    return -1


def _align(p, deg=None):
    """Left-align polynomial coefficients at word 17 of an 18-word array."""
    d = _poly_deg(p) if deg is None else deg
    arr = [0] * 18
    for i in range(d + 1):
        arr[17 - d + i] = p[i]
    return arr


def eea_eliminate(r0, r1, d0, d1, gf, max_iters=35):
    """Forward aligned elimination (NO parking in r0; quotients recorded).

    r0/r1 are 18-word left-aligned arrays.  Returns (r0, r1, qs, fires,
    d0, d1, a_r0) where a_r0 tracks the alignment exponent of r0
    (array == polynomial * z^a_r0).  The circuit parks each quotient in the
    zeroed syndrome words S36..S70 instead of the r0 array (35 slots >= the
    total iteration count), which avoids collision with the remainder data.
    """
    qs = []
    fires = []
    c = 0
    a_r0 = 17 - d0
    done = d1 < 0
    for _ in range(max_iters):
        q = gf.mul(r0[17], gf.inv(r1[17]))
        qs.append(q)
        for i in range(18):
            r0[i] ^= gf.mul(q, r1[i])
        r0 = [r0[-1]] + r0[:-1]
        c += 1
        a_r0 += 1
        fire = (not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0))
        if fire:
            r0, r1 = r1, r0
            d0, d1 = d1, d0 - c
            c = 0
            a_r0 = 17 - d0
            done = d1 < 0
        fires.append(fire)
    return r0, r1, qs, fires, d0, d1, a_r0


def eea_reconstruct(qs, fires, dE, dO, gf, max_iters=35, W=19):
    """Rebuilds (s0, s1, t0, t1) from the recorded quotients and fire pattern.

    19-word cyclic buffers (the true degree of t0 can reach 18).  Returns
    (s0, s1, t0, t1, a) with the array == poly * z^a.
    """
    s0 = [0] * W
    t0 = [0] * W
    s1 = [0] * W
    t1 = [0] * W
    s0[(17 - dE) % W] = 1
    t1[(17 - dO) % W] = 1
    c = 0
    a = 17 - dE
    dd0, dd1 = dE, dO
    done = dO < 0
    for it in range(max_iters):
        q = qs[it]
        for i in range(W):
            s0[i] ^= gf.mul(q, s1[i])
            t0[i] ^= gf.mul(q, t1[i])
        s0 = [s0[-1]] + s0[:-1]
        t0 = [t0[-1]] + t0[:-1]
        c += 1
        a += 1
        if fires[it]:
            s0, s1 = s1, s0
            t0, t1 = t1, t0
            dd0, dd1 = dd1, dd0 - c
            c = 0
            a = 17 - dd0
            done = dd1 < 0
    return s0, s1, t0, t1, a


def eea_uneliminate(r0, r1, qs, fires, gf, max_iters=35):
    """Exact mirror of eea_eliminate (quotients re-read from the parking)."""
    for it in range(max_iters - 1, -1, -1):
        if fires[it]:
            r0, r1 = r1, r0
        q = qs[it]
        r0 = r0[1:] + [r0[0]]          # rotate right (inverse of left rotate)
        for i in range(18):
            r0[i] ^= gf.mul(q, r1[i])
    return r0, r1


def eea_unreconstruct(s0, s1, t0, t1, qs, fires, gf, max_iters=35, W=19):
    """Exact mirror of eea_reconstruct."""
    for it in range(max_iters - 1, -1, -1):
        if fires[it]:
            s0, s1 = s1, s0
            t0, t1 = t1, t0
        q = qs[it]
        s0 = s0[1:] + [s0[0]]
        t0 = t0[1:] + [t0[0]]
        for i in range(W):
            s0[i] ^= gf.mul(q, s1[i])
            t0[i] ^= gf.mul(q, t1[i])
    return s0, s1, t0, t1


# ----------------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------------

def eval_poly(coeffs, point, gf):
    acc = 0
    for c in reversed(coeffs):
        acc = gf.mul(acc, point) ^ c
    return acc


def eval_sigma_from_eo(E, O, beta, gf):
    e = eval_poly(E, beta, gf)
    o = eval_poly(O, beta, gf)
    return gf.sq(e) ^ gf.mul(beta, gf.sq(o))


def eval_psi_from_ab(A, B, beta, gf):
    a = eval_poly(A, beta, gf)
    b = eval_poly(B, beta, gf)
    return gf.sq(b) ^ gf.mul(beta, gf.sq(a))


def bilinear_rank1(B):
    """Decompose the F2 bilinear form with matrix B into rank-1 terms."""
    n = B.shape[0]
    M = B.copy().astype(np.uint8)
    terms = []
    for r in range(n):
        piv = -1
        for k in range(n):
            if M[r, k]:
                piv = k
                break
        if piv < 0:
            continue
        u = np.zeros(n, dtype=np.uint8)
        u[r] = 1
        v = M[r].copy()
        terms.append((u, v))
        for r2 in range(n):
            if r2 != r and M[r2, piv]:
                M[r2] ^= M[r]
    return terms


# ----------------------------------------------------------------------------
# Full pipeline on one sample
# ----------------------------------------------------------------------------

def run_one(sample, gf, cutoffs, K_cut, n, mu_star, m):
    syndromes = sample['syndromes'][:]
    errors = sample['errors']
    mbuc = sample['mbuc']

    bm = BMState(gf, cutoffs, n=n)
    for r in range(1, n + 1):
        bm.round_fwd(r, syndromes)
    sigma = bm.sigma
    locator_deg = _poly_deg(sigma[:36])
    # roots of the locator vs the injected error positions
    g_inv = gf.inv(PRIMITIVE)
    roots_ok = True
    roots = []
    beta = 1
    for j in range(m):
        if j:
            beta = gf.mul(beta, g_inv)
        if eval_poly(sigma[:36], beta, gf) == 0:
            roots.append(j)
    roots_ok = sorted(roots) == sorted(errors.keys())

    # ---- normalize sigma so sigma0 == 1 --------------------------------
    sigma0_pre = sigma[0]
    delta = gf.inv(sigma0_pre)
    sigma = [gf.mul(sigma[i], delta) for i in range(36)]
    # (the circuit keeps sigma0_pre in a 12-bit keeper for the reverse)

    if K_cut < 35 and any(sigma[j] for j in range(K_cut, 36)):
        bm.fail_cutoff = True

    # ---- omega product (sigma0 == 1 now) -------------------------------
    omega_product(syndromes, sigma, K_cut, gf, reverse=False)
    omega_words = syndromes[:35]  # omega_0..omega_34
    tail_ok = all(x == 0 for x in syndromes[35:70])

    # ---- sqrt transform -------------------------------------------------
    E, O = sigma_to_eo(sigma, gf)

    # ---- EEA (aligned, fixed schedule) ----------------------------------
    r0 = _align(E)
    r1 = _align(O)
    d0, d1 = _poly_deg(E), _poly_deg(O)
    if d0 < d1:
        r0, r1 = r1, r0
        d0, d1 = d1, d0
    r0_init, r1_init = [x for x in r0], [x for x in r1]
    r0f, r1f, qs, fires, d0f, d1f, a_f = eea_eliminate(r0, r1, d0, d1, gf)
    s0f, s1f, t0f, t1f, a_s = eea_reconstruct(qs, fires, d0, d1, gf)
    # extraction: A = s0, B = t0; array == poly * z^a_s
    a_s %= 19
    Aw = [s0f[(a_s + i) % 19] for i in range(18)]
    Bw = [t0f[(a_s + i) % 19] for i in range(17)]
    c_gcd = r0f[a_f % 18]  # the constant gcd of (E, O)

    # Bezout check: A*E + B*O == c_gcd (constant)
    prod = [0] * 35
    for i in range(18):
        for j in range(18):
            if Aw[i] and j <= _poly_deg(E):
                prod[i + j] ^= gf.mul(Aw[i], E[j])
    for i in range(17):
        for j in range(18):
            if Bw[i] and j <= _poly_deg(O):
                prod[i + j] ^= gf.mul(Bw[i], O[j])
    bezout_ok = prod[0] == c_gcd and all(x == 0 for x in prod[1:])

    # ---- scan -----------------------------------------------------------
    # psi(beta) = (B(beta)^2 + beta*A(beta)^2)/c_gcd^2 ; fold 1/c_gcd^2 into
    # the classical multiplier mu_j (done offline in the circuit too).
    c_inv2 = gf.mul(gf.inv(c_gcd), gf.inv(c_gcd))
    phase = 0
    beta = 1
    gamma_inv = gf.inv(PRIMITIVE)
    for j in range(m):
        if j:
            beta = gf.mul(beta, gamma_inv)
        sig = eval_sigma_from_eo(E, O, beta, gf)
        if sig == 0:
            mu_j = 0
            mj = mbuc[j]
            for k in range(DEGREE):
                if (mj >> k) & 1:
                    mu_j ^= mu_star[k]
            mu_j = gf.mul(mu_j, c_inv2)
            om = eval_poly(omega_words, beta, gf)
            ps = eval_psi_from_ab(Aw, Bw, beta, gf)
            phase ^= gf.mul(gf.mul(mu_j, om), ps).bit_count() & 1

    # ---- reverse --------------------------------------------------------
    s0u, s1u, t0u, t1u = eea_unreconstruct(s0f, s1f, t0f, t1f, qs, fires, gf)
    r0u, r1u = eea_uneliminate(r0f, r1f, qs, fires, gf)
    eea_ok = (r0u == r0_init and r1u == r1_init and s1u == [0] * 19
              and t0u == [0] * 19)
    omega_product(syndromes, sigma, K_cut, gf, reverse=True)
    # un-normalize
    sigma = [gf.mul(sigma[i], sigma0_pre) for i in range(36)]
    bm.sigma = sigma
    for r in range(n, 0, -1):
        bm.round_rev(r, syndromes)

    syndromes_ok = syndromes == sample['syndromes']
    bm_ok = (bm.sigma == [1] + [0] * 35 and bm.tau == [1] + [0] * 35
             and bm.d == 0 and not bm.stack)
    phase_ok = (bool(phase) == sample['phase'])
    ok = (syndromes_ok and bm_ok and phase_ok and eea_ok and
          not bm.fail_stack and not bm.fail_cutoff)
    return dict(ok=ok, phase_ok=phase_ok, bezout_ok=bezout_ok,
                tail_ok=tail_ok, eea_ok=eea_ok, fail_stack=bm.fail_stack,
                fail_cutoff=bm.fail_cutoff, shift_top=bm.ev_shift_top,
                locator_deg=locator_deg, roots_ok=roots_ok,
                restored_deg=_poly_deg(sigma), synd_ok=syndromes_ok,
                bm_ok=bm_ok)


# ----------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------

C_MUL = 63
C_INV = 315
C_DIV = 378
C_CSWAP = 1
C_FEQ = 1


def estimate_cost(n=70, m=4095, cutoffs=None, K_cut=20):
    t = 0
    for r in range(1, n + 1):
        K = min(cutoffs[r - 1], r - 1)
        Km = min(cutoffs[r - 1], 35)
        t += (K + 1) * C_MUL * 2          # conv + clear
        t += C_MUL                       # alpha
        t += C_INV * 2                   # inv + del
        t += 12 * C_CSWAP                # select
        t += (Km + 1) * 12 * C_CSWAP     # conditional swap
        t += (Km + 1) * C_MUL            # MADD
        t += 12 + 19 + 24                # d update, stack, flags
    for r in range(1, n + 1):
        Km = min(cutoffs[r - 1], 35)
        t += C_INV                       # reverse multiplier (inv tau0 or pop inv)
        t += C_MUL * 3                   # m, del m, pop mult
        t += (Km + 1) * C_MUL            # R-MADD
        t += (Km + 1) * 12 * C_CSWAP     # un-swap
        t += 12 + 12 + 12 + 7
    prod_mults = sum(min(k - 1, K_cut) for k in range(2, n + 1))
    t += prod_mults * C_MUL * 2
    eea = 35 * (C_DIV + 18 * C_MUL + 40)      # elimination
    eea += 35 * (C_DIV + 18 * C_MUL + 12 + 220)  # re-run + CF
    t += eea * 2
    t += m * (12 + 12)                   # root test + 12 CCZ
    return t


def estimate_qubits():
    # peak during BM: syndromes 840 + sigma 432 + tau 432 + temps (3x12) +
    # flags (~8) + d (8) + stack (19) + inv scaffold (48)
    return 840 + 432 + 432 + 36 + 8 + 8 + 19 + 48


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

def run_experiment(num_samples=100000, seed=12345, n=70, m=4095,
                   cutoffs=None, K_cut=35):
    gf = GF12()
    dist = DQIDistribution(m=m, n=n, degree=DEGREE, r=2016, gf=gf)
    if cutoffs is None:
        cutoffs = [min(r, 35) for r in range(1, n + 1)]
    mu_star = gf.dual_basis()
    rng = np.random.default_rng(seed)
    n_fail = n_phase = n_bez = n_stack = n_cut = n_shifttop = n_tail = n_eea = 0
    n_roots_bad = 0
    degs = []
    t0 = time.time()
    for it in range(num_samples):
        s = dist.sample(rng)
        r = run_one(s, gf, cutoffs, K_cut, n, mu_star, m)
        degs.append(r['locator_deg'])
        if not r['ok']:
            n_fail += 1
        if not r['phase_ok']:
            n_phase += 1
        if not r['bezout_ok']:
            n_bez += 1
        if not r['tail_ok']:
            n_tail += 1
        if not r['eea_ok']:
            n_eea += 1
        if not r['roots_ok']:
            n_roots_bad += 1
        if r['fail_stack']:
            n_stack += 1
        if r['fail_cutoff']:
            n_cut += 1
        if r['shift_top']:
            n_shifttop += 1
    toff = estimate_cost(n=n, m=m, cutoffs=cutoffs, K_cut=K_cut)
    return dict(
        samples=num_samples, failures=n_fail, fail_rate=n_fail / num_samples,
        phase_mismatch=n_phase, bezout_mismatch=n_bez, tail_mismatch=n_tail,
        eea_mismatch=n_eea, roots_mismatch=n_roots_bad,
        stack_overflow=n_stack, cutoff_fail=n_cut, shift_top_events=n_shifttop,
        mean_locator_deg=float(np.mean(degs)), max_locator_deg=int(np.max(degs)),
        toffoli_estimate=toff, qubit_estimate=estimate_qubits(),
        mean_weight=dist.mean_weight,
        elapsed=time.time() - t0,
    )


if __name__ == '__main__':
    print('Smoke run: n=70, full cutoffs (deg<=35), 2000 samples...')
    res = run_experiment(num_samples=2000, seed=1, K_cut=35)
    for k, v in res.items():
        print(f'  {k}: {v}')
