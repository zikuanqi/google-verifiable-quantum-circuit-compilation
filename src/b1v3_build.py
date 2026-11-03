# -*- coding: utf-8 -*-
"""Benchmark I v3 circuit: centered signed-digit quadratic Horner, kickmix builder.

Independent implementation of the signed-digit architecture (see b1v3_model.py
for the arithmetic).  Building blocks:

  normalize      barrel-shift x << s into [2^21, 2^22) with 5 binary flags;
                 flags erased by recomputation or HMR + conditioned phase fixup.
  lookup         one kickmix QROM loads K2/K1/K0 (low and high words); the high
                 word is phased early and unloaded, the rest after the reverse
                 pass (measurement-based uncomputation).
  rrows / yrows  the two Horner sums; each row is one conditional add/subtract
                 of a shifted copy (complement trick turns the sign digit into
                 a subtraction), computed with a Gidney carry-ripple adder whose
                 carries are measurement-erased.
  phasing        the phase value is split into a high word (phased before the
                 rows, freeing its register) and a low word (phased after), each
                 barrel-shifted by (s-e)/2 and applied bitwise with z_pow.

The top x qubit is 1 for every valid input after normalization, so it is
borrowed as one extra clean ancilla for the carry chains (condition-aware
clean-qubit reuse).
"""
from __future__ import annotations

import argparse
from fractions import Fraction

import kickmix as km
import numpy as np

import b1v3_model as v3m

W = v3m.W
DB = v3m.DB
NSEG = v3m.NSEG
N_SHIFT_STAGES = ((4, 16), (3, 8), (2, 4), (1, 2), (0, 1))


# ---------------------------------------------------------------- primitives

def _and_into(b, controls, target):
    """target ^= AND(controls); transient ancillas measured away afterwards."""
    k = len(controls)
    if k == 1:
        b.cx(controls[0], target)
        return
    if k == 2:
        b.ccx(controls[0], controls[1], target)
        return
    anc = b.alloc_qubits(k - 2)
    b.init_and(controls[0], controls[1], anc[0])
    for j in range(1, k - 2):
        b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.ccx(anc[k - 3], controls[k - 1], target)
    for j in reversed(range(1, k - 2)):
        b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.del_and(controls[0], controls[1], anc[0])
    b.free(anc)


def _phase_and(b, controls):
    """Relative Z phase on the AND of `controls` (measurement-erased ancillas)."""
    k = len(controls)
    if k == 1:
        b.z(controls[0])
        return
    if k == 2:
        b.cz(controls[0], controls[1])
        return
    anc = b.alloc_qubits(k - 2)
    b.init_and(controls[0], controls[1], anc[0])
    for j in range(1, k - 2):
        b.init_and(anc[j - 1], controls[j + 1], anc[j])
    b.cz(anc[k - 3], controls[k - 1])
    for j in reversed(range(1, k - 2)):
        b.del_and(anc[j - 1], controls[j + 1], anc[j])
    b.del_and(controls[0], controls[1], anc[0])
    b.free(anc)


def _gadd(b, a, t, sub=False, cin=None, borrowed=()):
    """t += a + cin  (mod 2^len(t)), Gidney adder, carries measurement-erased.

    len(a) <= len(t).  `borrowed` are extra clean qubits used for the carry
    chain instead of fresh allocations.  The top carry is folded directly into
    t[-1] (no full-width carry qubit).
    """
    a, t = list(a), list(t)
    o, n = len(a), len(t)
    assert o <= n and n >= 1
    if sub:
        for q in t:
            b.x(q)
    if n == 1:
        if o:
            b.cx(a[0], t[0])
        if cin is not None:
            b.cx(cin, t[0])
        if sub:
            for q in t:
                b.x(q)
        return
    taken = list(borrowed)[: max(0, n - 2)]
    need = max(0, n - 2 - len(taken))
    fresh = b.alloc_qubits(need) if need else km.array([])
    c = taken + list(fresh)

    def carry(i):
        return cin if i == 0 else c[i - 1]

    # forward: ripple the carries, XOR the operand into the low bits.
    for i in range(n - 2):
        if i == 0 and cin is None:
            if o:
                b.init_and(a[0], t[0], carry(1))
        elif i < o:
            b.cx(carry(i), a[i])
            b.cx(carry(i), t[i])
            b.init_and(a[i], t[i], carry(i + 1))
            b.cx(carry(i), carry(i + 1))
        else:
            b.init_and(carry(i), t[i], carry(i + 1))
    # fold the final carry into t[-1]
    i = n - 2
    top = carry(i)
    if top is None:
        if i < o:
            b.ccx(a[i], t[i], t[-1])
            b.cx(a[i], t[i])
    elif i < o:
        b.cx(top, a[i])
        b.cx(top, t[i])
        b.ccx(a[i], t[i], t[-1])
        b.cx(top, t[-1])
        b.cx(top, a[i])
        b.cx(a[i], t[i])
    else:
        b.ccx(top, t[i], t[-1])
        b.cx(top, t[i])
    if n - 1 < o:
        b.cx(a[n - 1], t[n - 1])
    # reverse: erase the carries by measurement.
    for i in reversed(range(n - 2)):
        if i == 0 and cin is None:
            if o:
                b.del_and(a[0], t[0], carry(1))
                b.cx(a[0], t[0])
        elif i < o:
            b.cx(carry(i), carry(i + 1))
            b.del_and(a[i], t[i], carry(i + 1))
            b.cx(carry(i), a[i])
            b.cx(a[i], t[i])
        else:
            b.del_and(carry(i), t[i], carry(i + 1))
            b.cx(carry(i), t[i])
    if need:
        b.free(fresh)
    if sub:
        for q in t:
            b.x(q)


# ------------------------------------------------------------ normalization

def _normalize(b, x, flags):
    """x <<= s until x[21]=1, flags hold s in binary (5 bits)."""
    for k, d in N_SHIFT_STAGES:
        top = x[W - d:W]
        if d == 1:
            b.cx(x[W - 1], flags[0])
            b.x(flags[0])                      # flag = NOT x[21]
        else:
            b.x(top)
            _and_into(b, list(top), flags[k])
            b.x(top)
        for i in reversed(range(0, W - d)):
            b.cswap(flags[k], x[i], x[i + d])


def _unnormalize(b, x, flags):
    """Inverse of _normalize; flags erased (recompute for bit 0, HMR + fixup else)."""
    for k, d in reversed(N_SHIFT_STAGES):
        for i in range(0, W - d):
            b.cswap(flags[k], x[i], x[i + d])
        top = x[W - d:W]
        if d == 1:
            b.x(flags[0])
            b.cx(x[W - 1], flags[0])
        else:
            m = b.alloc_bits(1)[0]
            b.hmr(flags[k], m)
            b.push_condition(m)                # fixup on the shots where m == 1
            b.x(top)
            _phase_and(b, list(top))
            b.x(top)
            b.pop_condition()
            b.free(m)


# ------------------------------------------------------------------ phasing

def _shift_word(b, flags, word, reg_len, inverse):
    """Barrel-shift `word` left by (s - flags[0]) / 2, using flags[1..4]."""
    stages = ((1, 1), (2, 2), (3, 4), (4, 8))
    occ = []
    o = reg_len
    for _, d in stages:
        occ.append(o)
        o = min(o + d, len(word))
    if not inverse:
        for idx, (k, d) in enumerate(stages):
            for i in reversed(range(0, occ[idx])):
                if i + d < len(word):
                    b.cswap(flags[k], word[i], word[i + d])
    else:
        for idx, (k, d) in reversed(list(enumerate(stages))):
            for i in range(0, occ[idx]):
                if i + d < len(word):
                    b.cswap(flags[k], word[i], word[i + d])


def _phase(b, flags, reg, base_bit, fy):
    """Apply the phase of `reg` (shifted by (s-e)/2) bitwise with z_pow."""
    ext = b.alloc_qubits(10)
    word = km.array(list(reg) + list(ext))
    _shift_word(b, flags, word, len(reg), inverse=False)
    for j in range(len(word)):
        exponent = base_bit + j - fy - 10
        if exponent < 0:
            b.z_pow(word[j], Fraction(2) ** (exponent + 1))
    _shift_word(b, flags, word, len(reg), inverse=True)
    b.free(ext)


# ------------------------------------------------------------- Horner stages

def _rrows(b, cfg, delta, c2, r, inverse, w1, adder=_gadd):
    """R += sig_i * round(K2 / 2^sh2_i) for the kept rows (or -= for inverse)."""
    seq = cfg.rrows()
    if inverse:
        seq = list(reversed(seq))
    for i in seq:
        sh = cfg.sh2(i)
        if sh >= 0:
            off = c2[sh:cfg.w2]
            t = r[:w1]
            cin = c2[sh - 1] if sh >= 1 else None
        else:
            off = c2[:cfg.w2]
            t = r[-sh:w1]
            cin = None
        b.cx(delta[i], t)
        adder(b, off, t, sub=inverse, cin=cin)
        b.cx(delta[i], t)


def _yrows(b, cfg, delta, r, lr, inverse, w1, adder=_gadd):
    """Lr -= sig_i * round(R / 2^sh1_i) for all DB rows (or += for inverse)."""
    seq = list(reversed(range(DB)))
    if inverse:
        seq.reverse()
    for i in seq:
        sh = cfg.sh1(i)
        if sh >= 0:
            cin = r[sh - 1] if sh >= 1 else None
            off = r[sh:w1]
            t = lr
        else:
            off = r[:w1]
            t = lr[-sh:]
            cin = None
        b.cx(delta[i], t)
        adder(b, off, t, sub=inverse, cin=cin)
        b.cx(delta[i], t)


# --------------------------------------------------------------------- build

def build_v3_circuit(cfg: v3m.V3Cfg, tabs: dict, lay: dict) -> km.Circuit:
    w1, lb, s, whi = lay["w1"], cfg.lb, cfg.s, lay["whi"]
    w2 = cfg.w2
    b = km.CircuitBuilder()
    x = b.create_quantum_register(W, name="x")
    flags = b.alloc_qubits(5)
    with b.mark("normalize"):
        _normalize(b, x, flags)
    # x[21] == 1 for every valid input: borrow it as a clean ancilla.
    b.x(x[-1])
    c2 = b.alloc_qubits(w2)
    r = b.alloc_qubits(w1)
    lr = b.alloc_qubits(lb)
    yh = b.alloc_qubits(whi)
    remaining = km.array(list(lr) + list(r) + list(c2))
    target = km.array(list(remaining) + list(yh))

    full = [0] * 256
    rest = [0] * 256
    high = [0] * 256
    for (e, a), (k2, k1, k0) in tabs.items():
        idx = a + NSEG * e
        rest[idx] = lay["lo"][(e, a)] | (k1 << lb) | (k2 << (lb + w1))
        high[idx] = lay["hi"][(e, a)]
        full[idx] = rest[idx] | (high[idx] << (lb + w1 + w2))
    address = km.array(list(x[14:21]) + [flags[0]])
    delta = x[:DB]

    adder = lambda bb, a, t, sub=False, cin=None: _gadd(bb, a, t, sub, cin, borrowed=(x[-1],))

    with b.mark("lookup"):
        b.init_lookup(table=full, address=address, target=target)
    with b.mark("phase_high"):
        _phase(b, flags, yh, s, cfg.fy)
    with b.mark("del_high"):
        b.del_lookup(table=high, address=address, target=yh)
    b.free(yh)
    with b.mark("rrows"):
        _rrows(b, cfg, delta, c2, r, False, w1, adder)
    with b.mark("yrows"):
        _yrows(b, cfg, delta, r, lr, False, w1, adder)
    # low word: bits GY..LB-1 of K0 - Q (bit S participates in the seam carry)
    with b.mark("phase_low"):
        _phase(b, flags, lr[cfg.gy:lb], cfg.gy, cfg.fy)
    with b.mark("yrows_rev"):
        _yrows(b, cfg, delta, r, lr, True, w1, adder)
    with b.mark("rrows_rev"):
        _rrows(b, cfg, delta, c2, r, True, w1, adder)
    with b.mark("del_rest"):
        b.del_lookup(table=rest, address=address, target=remaining)
    b.free(remaining)
    b.x(x[-1])
    with b.mark("unnormalize"):
        _unnormalize(b, x, flags)
    b.free(flags)
    return b.finish_circuit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Path of the .kmb file to write.")
    parser.add_argument("--tabs", help=".npz with table (256,3) [k2,k1,k0].")
    parser.add_argument("--f2", type=int, default=8)
    parser.add_argument("--f1", type=int, default=19)
    parser.add_argument("--fyr", type=int, default=27)
    parser.add_argument("--gy", type=int, default=1)
    parser.add_argument("--drop2", type=str, default="5")
    args = parser.parse_args()
    drop2 = tuple(int(t) for t in args.drop2.split(",")) if args.drop2 else ()
    cfg = v3m.V3Cfg(f2=args.f2, f1=args.f1, fyr=args.fyr, gy=args.gy, drop2=drop2)
    if args.tabs:
        tab = np.load(args.tabs)["table"]
        tabs = {(e, a): tuple(int(v) for v in tab[a + NSEG * e])
                for e in (0, 1) for a in range(NSEG)}
    else:
        print("fitting coefficients ...")
        tabs, _ = v3m.fit_all(cfg)
        np.savez(args.output + ".tabs.npz",
                 table=np.array([tabs[(e, a)] for e in (0, 1) for a in range(NSEG)],
                                dtype=np.int64),
                 config=np.array([args.f2, args.f1, args.fyr, args.gy]),
                 drop=np.array(list(drop2)))
    lay = v3m.make_layout(cfg, tabs)
    circuit = build_v3_circuit(cfg, tabs, lay)
    circuit.to_file(args.output, format="kmb")
    st = circuit.operation_statistics()
    print(f"Wrote {args.output}: {st.num_toffoli_like:,} Toffolis, "
          f"{circuit.num_qubits} qubits, {len(circuit):,} ops")


if __name__ == "__main__":
    main()
