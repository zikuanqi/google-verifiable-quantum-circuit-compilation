"""b2_build.py — kickmix circuit builder for the reversible Berlekamp-Massey
Reed-Solomon MBUC decoder (Benchmark II DQI), per Gu et al. arXiv:2610.22367.

Mirrors the register-level protocol validated in src/b2_model.py:

  1. Forward BM (n rounds): rescaled updates, shared MADD via conditional
     routing (CSWAP under the replace flag), fused-Q shift stack: a push of
     (tag, delta) fires on EVERY round with d < 0 (pre-update); overflow
     (drop) is the declared-failure class.
  2. Reverse BM: branch detection replace = d7 AND NOT (top tag == r) after
     undoing the +1 (the reference's exact SGN scheme; no coincidence
     conjuncts); the discrepancy comes back either from the stack (rt) or
     from the scaled identity sigma0*delta == 1 (rep).
  3. Normalize sigma (sigma0 -> 1), keep sigma0_pre in a keeper.
  4. In-place omega product over all n syndrome words (descending order).
  5. sqrt permutation + linear maps: sigma -> (E, O) at sigma[0..17]/[18..35].
  6. Aligned fixed-schedule EEA (35 iterations) with quotients parked in the
     zeroed syndrome words S[L+1..n-1], then a 2-pass continued-fraction
     reconstruction of (A, B) (19-word cyclic buffers; gcd kept in a keeper).
  7. Chien scan over all m positions: root test (12 T) + bilinear phase via
     rank-1 decomposition of Tr(mu_j * x * y) with 12 CCZ gates; mu_j folds
     the classical mask and the constant 1/c_gcd^2.
  8. Exact mirrors of 3-6 and the full restoration.

Registers: exactly S0..S{n-1} (quantum) and E0..E{m-1} (classical), all width
`degree`.  Everything else is anonymous alloc_qubits/alloc_bits.
"""

from __future__ import annotations

import argparse

import kickmix as km


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _not(builder, q):
    """t = not q into a fresh qubit."""
    t = builder.alloc_qubits(1)[0]
    builder.cx(q, t)
    builder.x(t)
    return t


def _and(builder, a, b):
    t = builder.alloc_qubits(1)
    builder.init_and(a, b, t)
    return t[0]


def _or(builder, a, b):
    """t = a or b into a fresh qubit (clear via _unor)."""
    t = builder.alloc_qubits(1)[0]
    builder.cx(a, t)
    builder.cx(b, t)
    builder.ccx(a, b, t)
    return t


def _unor(builder, a, b, t):
    builder.ccx(a, b, t)
    builder.cx(b, t)
    builder.cx(a, t)


def _cond_clear(builder, qbits, control):
    """Clear each qubit in qbits under control: q ^= (control and q)."""
    for q in qbits:
        builder.ccx(control, q, q)


def _xor_imm_under(builder, qbits, value, control):
    """qbits ^= value when control (value is a classical int)."""
    for i, q in enumerate(qbits):
        if (value >> i) & 1:
            builder.cx(control, q)


def _assign(builder, dst, value, control):
    """dst := value under control; dst must have been 0 or xor'd cleanly."""
    _xor_imm_under(builder, dst, value, control)


def bitlist(v, width):
    return [(v >> bp) & 1 for bp in range(width)]


# ----------------------------------------------------------------------------
# Berlekamp-Massey (rescaled, register-level)
# ----------------------------------------------------------------------------

class BMRegs:
    """Register file for the rescaled BM.

    The stack follows the reference's fused-Q scheme: a CAPS-slot shift
    stack of (tag, delta) entries; a push fires on EVERY round with d < 0
    (pre-update), so on the reverse the branch bit is
        rep = SGN(d) AND NOT (top tag == r)
    where d is the post-(+1)-undo (complemented) counter -- an exact
    detection with no coincidence conjuncts.  Overflow (drop) is the
    declared-failure class.
    """

    def __init__(self, builder, n, degree, caps=2):
        self.n = n
        self.L = n // 2
        self.degree = degree
        self.caps = caps
        self.tagw = (n - 1).bit_length()
        self.sigma = [builder.alloc_qubits(degree) for _ in range(self.L + 1)]
        self.tau = [builder.alloc_qubits(degree) for _ in range(self.L + 1)]
        self.d = builder.alloc_qubits(8)
        self.stk = [builder.alloc_qubits(degree + self.tagw) for _ in range(caps)]
        # classical MBU outcome bits (kept for the reverse-pass neg corrections)
        self.c_delta = [builder.alloc_bits(degree) for _ in range(n)]
        self.c_alpha = [builder.alloc_bits(degree) for _ in range(n)]
        # reverse-pass HMR outcomes (sigma0 keeper and the rep flag)
        self.c_keep = [builder.alloc_bits(degree) for _ in range(n)]
        self.c_rep = [builder.alloc_bits(1) for _ in range(n)]
        # init: sigma0 = tau0 = 1
        builder.x(self.sigma[0][0])
        builder.x(self.tau[0][0])


def _neg_correct(builder, reg, cbits):
    """Apply phase (-1)^{<c, reg>} via Z gates conditioned on classical bits."""
    for i in range(len(reg)):
        builder.push_condition(cbits[i])
        builder.z(reg[i])
        builder.pop_condition()


def bm_round_fwd(builder, bm, gf, syndromes, r, K_r):
    """Forward BM round r (1-based).  K_r = active sigma range cap."""
    L = bm.L
    degree = bm.degree
    sigma, tau = bm.sigma, bm.tau
    K = min(K_r, r - 1)
    Km = min(K_r, L)
    stk = bm.stk
    tagw = bm.tagw

    # 1. shift tau: cyclic left rotate by one word (0 Toffoli)
    for bit in range(degree):
        lane = km.array([tau[w][bit] for w in range(L + 1)])
        builder.left_rotate(lane, 1)

    # 2. discrepancy delta = sigma0*S_r + sum sigma_i * S_{r-i}
    delta_t = builder.alloc_qubits(degree)
    builder.ixor_gf2_mul(sigma[0], syndromes[r - 1], target=delta_t, field=gf)
    for i in range(1, K + 1):
        builder.ixor_gf2_mul(sigma[i], syndromes[r - 1 - i], target=delta_t,
                             field=gf)

    # 3. flags: eq0 = (delta == 0); nz = not eq0; dn = not d7;
    #    rep = nz and dn
    eq0 = builder.alloc_qubits(1)[0]
    builder.flip_if_equal(delta_t, 0, target=eq0)
    nz = _not(builder, eq0)
    dn = _not(builder, bm.d[7])          # d >= 0
    rep = _and(builder, nz, dn)
    nrep = _not(builder, rep)

    # 4. alpha = delta * sigma0 ; alpha_inv = inv(alpha)
    alpha_t = builder.alloc_qubits(degree)
    builder.init_gf2_mul(delta_t, sigma[0], target=alpha_t, field=gf)
    alpha_inv, inv_scaff = builder.init_gf2_inverse_with_scaffold(
        alpha_t, field=gf)

    # 5. conditional routing: swap sigma <-> tau under rep
    for i in range(Km + 1):
        builder.cswap(rep, sigma[i], tau[i])

    # 6. shared MADD as a gated double: sigma ^= alpha*tau (retain) or
    #    alpha_inv*tau (replace)
    for i in range(Km + 1):
        builder.ixor_gf2_mul(alpha_t, tau[i], target=sigma[i], field=gf,
                             control=nrep)
        builder.ixor_gf2_mul(alpha_inv, tau[i], target=sigma[i], field=gf,
                             control=rep)

    # 7. alpha clears: del-with-scaffold, then MBU-erase by HMR
    builder.del_gf2_inverse_with_scaffold(alpha_t, target=alpha_inv,
                                          scaffold=inv_scaff, field=gf)
    builder.free(inv_scaff)
    builder.free(alpha_inv)
    builder.hmr(alpha_t, bm.c_alpha[r - 1])
    builder.free(alpha_t)

    # 8. clear nrep (mirror of its creation)
    builder.cx(rep, nrep)
    builder.x(nrep)
    builder.free(nrep)

    # 9. dn2 := dn (copy of the pre-update sign flag; dn itself must later
    #    mirror d7_new, so rep's clear uses this copy)
    dn2 = builder.alloc_qubits(1)[0]
    builder.cx(dn, dn2)

    # 10. clear eq0 while delta_t still holds delta (pre-push)
    builder.flip_if_equal(delta_t, 0, target=eq0)

    # 11. fused-Q push on every d < 0 round (control = d7, pre-update);
    #     delta_t swaps its delta into the stack (and becomes 0 on correct
    #     paths, since the vacated slot is empty)
    for i in range(bm.caps - 2, -1, -1):
        builder.cswap(bm.d[7], stk[i], stk[i + 1])
    for i in range(degree):
        builder.cswap(bm.d[7], delta_t[i], stk[0][i])
    for bp, on in enumerate(bitlist(r, tagw)):
        if on:
            builder.cx(bm.d[7], stk[0][degree + bp])

    # 12. d := -d if rep (the complement); d7 = d7_new
    builder.cx(rep, bm.d)

    # 13. dn -> 1  (dn = not(d7_pre) ^ d7_new ^ rep = 1)
    builder.cx(bm.d[7], dn)
    builder.cx(rep, dn)

    # 14. clear rep: rep ^= nz * dn2 == nz * not(d7_pre) == rep
    builder.del_and(nz, dn2, rep)
    builder.free(rep)

    # 15-16. clear dn2 = not(d7_pre): dn2 ^= dn(=1) -> d7_pre,
    #         dn2 ^= d7_new -> d7_pre ^ d7_new = rep = 0
    builder.cx(dn, dn2)
    builder.cx(bm.d[7], dn2)
    builder.free(dn2)

    # 17. clear dn
    builder.x(dn)
    builder.free(dn)

    # 18. clear nz (eq0 still alive and already cleared)
    builder.cx(eq0, nz)
    builder.x(nz)
    builder.free(nz)
    builder.free(eq0)

    # 19. the +1 (completes d := -d or d := d+1)
    builder.iadd(1, target=bm.d)

    # 20. MBU-erase delta by HMR (post-push value: 0 on push rounds, delta
    #     otherwise; the reverse pass matches this exactly)
    builder.hmr(delta_t, bm.c_delta[r - 1])
    builder.free(delta_t)


def bm_round_rev(builder, bm, gf, syndromes, r, K_r):
    """Reverse BM round r (1-based)."""
    L = bm.L
    degree = bm.degree
    sigma, tau = bm.sigma, bm.tau
    Km = min(K_r, L)
    K = min(K_r, r - 1)
    stk = bm.stk
    tagw = bm.tagw

    # 1. undo the +1: d = the complemented value (sign = rep OR pushed)
    builder.isub(1, target=bm.d)

    # 2. branch detection: rt = (top tag == r); rep = d7 AND NOT rt
    rt = builder.alloc_qubits(1)[0]
    builder.flip_if_equal(stk[0][degree:], r, target=rt)
    rep = builder.alloc_qubits(1)[0]
    builder.cx(bm.d[7], rep)
    builder.ccx(bm.d[7], rt, rep)
    nrep = _not(builder, rep)

    # 3. rt2 := d7 AND nrep (equals rt on correct paths; survives rt's clear)
    rt2 = _and(builder, bm.d[7], nrep)

    # 4. clear rt while the top tag is still the original (pre-shift)
    builder.flip_if_equal(stk[0][degree:], r, target=rt)
    builder.free(rt)

    # 5. un-tag under rt2 (mirror of the forward tag write)
    _xor_imm_under(builder, stk[0][degree:], r, rt2)

    # 6. pop the value under rt2 (mirror of the forward push), then un-shift
    delta_t = builder.alloc_qubits(degree)
    for i in range(degree):
        builder.cswap(rt2, delta_t[i], stk[0][i])
    for i in range(bm.caps - 1):
        builder.cswap(rt2, stk[i], stk[i + 1])
    builder.del_and(bm.d[7], nrep, rt2)
    builder.free(rt2)

    # 7. delta_inv := inv(sigma0_post); under rep, delta_t ^= delta_inv turns
    #    delta_t into inv(sigma0_post) == delta (the scaled identity).  The
    #    scaffold is kept alive until step 13: its del needs sigma0_post,
    #    which the MADD would destroy -- hence the keeper at step 8.
    delta_inv, dinv_scaff = builder.init_gf2_inverse_with_scaffold(
        sigma[0], field=gf)
    for i in range(degree):
        builder.ccx(rep, delta_inv[i], delta_t[i])

    # 8. keep sigma0_post (for the step-13 del)
    s0_keep = builder.alloc_qubits(degree)
    for i in range(degree):
        builder.cx(sigma[0][i], s0_keep[i])

    # 9. multipliers: alpha_a = delta'*sigma0 (retain m), alpha_b = delta'*tau0
    #    (forward alpha), binv = inv(alpha_b) (replace m).
    alpha_a = builder.alloc_qubits(degree)
    builder.init_gf2_mul(delta_t, sigma[0], target=alpha_a, field=gf)
    alpha_b = builder.alloc_qubits(degree)
    builder.init_gf2_mul(delta_t, tau[0], target=alpha_b, field=gf)
    binv, bscaff = builder.init_gf2_inverse_with_scaffold(alpha_b, field=gf)

    # 10. reverse MADD (gated double)
    for i in range(Km + 1):
        builder.ixor_gf2_mul(alpha_a, tau[i], target=sigma[i], field=gf,
                             control=nrep)
        builder.ixor_gf2_mul(binv, tau[i], target=sigma[i], field=gf,
                             control=rep)

    # 11. neg corrections for the forward MBU kickbacks:
    #   <c_delta, delta> unconditional; <c_alpha, alpha> per branch.
    _neg_correct(builder, delta_t, bm.c_delta[r - 1])
    for i in range(degree):
        builder.push_condition(bm.c_alpha[r - 1][i])
        builder.cz(nrep, alpha_a[i])
        builder.cz(rep, alpha_b[i])
        builder.pop_condition()

    # 12. clear the multiplier temps (operands still valid here)
    builder.del_gf2_inverse_with_scaffold(alpha_b, target=binv,
                                          scaffold=bscaff, field=gf)
    builder.free(bscaff)
    builder.free(binv)
    builder.del_gf2_mul(delta_t, tau[0], target=alpha_b, field=gf)
    builder.free(alpha_b)
    builder.del_gf2_mul(delta_t, sigma[0], target=alpha_a, field=gf)
    builder.free(alpha_a)

    # 13. delta' clear, replace branch: delta_t ^= delta_inv == delta
    for i in range(degree):
        builder.ccx(rep, delta_inv[i], delta_t[i])
    builder.del_gf2_inverse_with_scaffold(s0_keep, target=delta_inv,
                                          scaffold=dinv_scaff, field=gf)
    builder.free(dinv_scaff)
    builder.free(delta_inv)

    # 14. delta' clear, retain/no-op branch: reverse convolution with the
    #     restored sigma, gated on nrep (on rep rounds sigma is the shifted
    #     tau, whose convolution is not the discrepancy)
    builder.ixor_gf2_mul(sigma[0], syndromes[r - 1], target=delta_t,
                         field=gf, control=nrep)
    for i in range(1, K + 1):
        builder.ixor_gf2_mul(sigma[i], syndromes[r - 1 - i], target=delta_t,
                             field=gf, control=nrep)
    builder.free(delta_t)

    # 15. erase the keeper (operand-only register: no kickbacks target it,
    #     so a plain HMR is phase-free)
    builder.hmr(s0_keep, bm.c_keep[r - 1])
    builder.free(s0_keep)

    # 16. un-swap
    for i in range(Km + 1):
        builder.cswap(rep, sigma[i], tau[i])

    # 17. restore the pre-round d: undo the complement under rep
    builder.cx(rep, bm.d)

    # 18. clear nrep (mirror of its creation)
    builder.cx(rep, nrep)
    builder.x(nrep)
    builder.free(nrep)

    # 19. erase rep (a control-only flag: nothing kicks onto it, so a plain
    #     HMR is phase-free)
    builder.hmr([rep], bm.c_rep[r - 1])
    builder.free(rep)

    # 20. un-shift tau (inverse cyclic rotate)
    for bit in range(degree):
        lane = km.array([tau[w][bit] for w in range(L + 1)])
        builder.right_rotate(lane, 1)


# ----------------------------------------------------------------------------
# Normalization, product, sqrt transform
# ----------------------------------------------------------------------------

def normalize_sigma(builder, sigma, sigma0_keep, gf):
    """sigma := sigma * inv(sigma0); keeps sigma0_pre in sigma0_keep."""
    L1 = len(sigma) - 1
    degree = len(sigma[0])
    # delta = inv(sigma0)
    delta, dscaff = builder.init_gf2_inverse_with_scaffold(sigma[0], field=gf)
    # keep sigma0_pre
    for i in range(degree):
        builder.cx(sigma[0][i], sigma0_keep[i])
    for w in range(L1 + 1):
        t = builder.alloc_qubits(degree)
        builder.init_gf2_mul(delta, sigma[w], target=t, field=gf)
        builder.cx(t, sigma[w])
        builder.cx(sigma[w], t)
        builder.cx(t, sigma[w])
        builder.ixor_gf2_mul(sigma0_keep, sigma[w], target=t, field=gf)
        builder.free(t)
    builder.del_gf2_inverse_with_scaffold(sigma[0], target=delta,
                                          scaffold=dscaff, field=gf)
    builder.free(dscaff)
    builder.free(delta)


def unnormalize_sigma(builder, sigma, sigma0_keep, gf):
    """sigma := sigma * sigma0_pre (reverse of normalize_sigma)."""
    L1 = len(sigma) - 1
    degree = len(sigma[0])
    delta, dscaff = builder.init_gf2_inverse_with_scaffold(sigma0_keep, field=gf)
    for w in range(L1 + 1):
        t = builder.alloc_qubits(degree)
        builder.init_gf2_mul(sigma0_keep, sigma[w], target=t, field=gf)
        builder.cx(t, sigma[w])
        builder.cx(sigma[w], t)
        builder.cx(t, sigma[w])
        builder.ixor_gf2_mul(delta, sigma[w], target=t, field=gf)
        builder.free(t)
    builder.del_gf2_inverse_with_scaffold(sigma0_keep, target=delta,
                                          scaffold=dscaff, field=gf)
    builder.free(dscaff)
    builder.free(delta)


def omega_product(builder, syndromes, sigma, K_cut, gf, *, reverse=False):
    n = len(syndromes)
    if reverse:
        for k in range(1, n):
            for j in range(1, min(k, K_cut) + 1):
                builder.ixor_gf2_mul(sigma[j], syndromes[k - j],
                                     target=syndromes[k], field=gf)
    else:
        for k in range(n - 1, 0, -1):
            for j in range(1, min(k, K_cut) + 1):
                builder.ixor_gf2_mul(sigma[j], syndromes[k - j],
                                     target=syndromes[k], field=gf)


# ----------------------------------------------------------------------------
# Linear maps over F2 (square / sqrt of a field element, in place)
# ----------------------------------------------------------------------------

def _linear_map_inplace(builder, target, mat):
    """Apply the F2-linear map (mat: list of row masks, little endian) to the
    little-endian register `target`, in place, CNOT-only.  Uses a fresh
    scratch register y := M*x, then a 3-CX swap dance, then y ^= M*x again
    to clear the scratch."""
    n = len(target)
    y = builder.alloc_qubits(n)

    def layer(src, dst):
        for b in range(n):
            parity = [src[k] for k in range(n) if (mat[b] >> k) & 1]
            if parity:
                builder.cx(parity, dst[b])

    layer(target, y)
    builder.cx(y, target)
    builder.cx(target, y)
    builder.cx(y, target)
    layer(target, y)
    builder.free(y)


def _sq_mat(gf):
    d = gf.degree
    mat = [0] * d
    for k in range(d):
        v = gf.mul(1 << k, 1 << k)
        for b in range(d):
            if (v >> b) & 1:
                mat[b] |= 1 << k
    return mat


def _sqrt_mat(gf):
    # inverse of the squaring map
    d = gf.degree
    sqm = _sq_mat(gf)
    # invert the F2 matrix
    A = [[(sqm[r] >> c) & 1 for c in range(d)] for r in range(d)]
    A = [row + [1 if i == r else 0 for i in range(d)] for r, row in enumerate(A)]
    for col in range(d):
        piv = next(r for r in range(col, d) if A[r][col])
        A[col], A[piv] = A[piv], A[col]
        for r in range(d):
            if r != col and A[r][col]:
                A[r] = [x ^ y for x, y in zip(A[r], A[col])]
    inv = [0] * d
    for r in range(d):
        for c in range(d):
            if A[r][d + c]:
                inv[c] |= 1 << r
    return inv


def sqrt_transform(builder, sigma, gf, *, reverse=False):
    """sigma -> (E, O): E_i = sqrt(sigma_{2i}) at sigma[0..17],
    O_i = sqrt(sigma_{2i+1}) at sigma[18..35].  reverse=True undoes it.

    The permutation word i <- word (2i mod L) for i = 1..L-1 (L = n//2) is
    decomposed into cycles and realized as chains of adjacent SWAPs
    (0 Toffoli)."""
    L1 = len(sigma) - 1          # == n//2
    L = L1
    sqrtm = _sqrt_mat(gf)
    sqm = _sq_mat(gf)
    # permutation word i <- word (2i mod L) for i = 1..L-1 (words 0 and L fixed)
    seen = [False] * L
    cycles = []
    for i in range(1, L):
        if seen[i]:
            continue
        cyc = []
        x = i
        while not seen[x]:
            seen[x] = True
            cyc.append(x)
            x = (2 * x) % L
        if len(cyc) > 1:
            cycles.append(cyc)
    mat = sqrtm if not reverse else sqm
    if not reverse:
        for cyc in cycles:
            for j in range(len(cyc) - 1):
                builder.swap(sigma[cyc[j]], sigma[cyc[j + 1]])
    else:
        for cyc in reversed(cycles):
            for j in range(len(cyc) - 2, -1, -1):
                builder.swap(sigma[cyc[j]], sigma[cyc[j + 1]])
    for w in range(L1 + 1):
        _linear_map_inplace(builder, sigma[w], mat)


# ----------------------------------------------------------------------------
# Main build
# ----------------------------------------------------------------------------

def build_circuit(m=4095, n=70, degree=12):
    gf = km.GF2Field(degree)
    L = n // 2
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}')
                 for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f'E{j}')
                  for j in range(m)]

    bm = BMRegs(builder, n, degree)
    sigma0_keep = builder.alloc_qubits(degree)
    gcd_keep = builder.alloc_qubits(degree)

    K_r = [min(r, L) for r in range(1, n + 1)]

    with builder.mark('forward_bm'):
        for r in range(1, n + 1):
            bm_round_fwd(builder, bm, gf, syndromes, r, K_r[r - 1])

    with builder.mark('normalize'):
        normalize_sigma(builder, bm.sigma, sigma0_keep, gf)

    with builder.mark('product'):
        omega_product(builder, syndromes, bm.sigma, L, gf, reverse=False)

    with builder.mark('sqrt'):
        sqrt_transform(builder, bm.sigma, gf, reverse=False)

    # ---- EEA + scan to be added (next stage) ------------------------------
    with builder.mark('placeholder_eea_scan'):
        pass

    with builder.mark('un_sqrt'):
        sqrt_transform(builder, bm.sigma, gf, reverse=True)

    with builder.mark('un_product'):
        omega_product(builder, syndromes, bm.sigma, L, gf, reverse=True)

    with builder.mark('unnormalize'):
        unnormalize_sigma(builder, bm.sigma, sigma0_keep, gf)

    with builder.mark('reverse_bm'):
        for r in range(n, 0, -1):
            bm_round_rev(builder, bm, gf, syndromes, r, K_r[r - 1])

    builder.free(sigma0_keep)
    builder.free(gcd_keep)
    for s in bm.stk:
        builder.free(s)
    builder.free(bm.d)
    for w in bm.tau:
        builder.free(w)
    for w in bm.sigma:
        builder.free(w)
    return builder.finish_circuit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', help='output .kmb path')
    parser.add_argument('--m', type=int, default=4095)
    parser.add_argument('--n', type=int, default=70)
    parser.add_argument('--degree', type=int, default=12)
    args = parser.parse_args()
    circuit = build_circuit(args.m, args.n, args.degree)
    circuit.to_file(args.output, format='kmb')
    st = circuit.operation_statistics()
    print(f'Wrote {args.output}: {st.num_toffoli_like:,} Toffolis, '
          f'{circuit.num_qubits} qubits, {len(circuit):,} ops')


if __name__ == '__main__':
    main()
