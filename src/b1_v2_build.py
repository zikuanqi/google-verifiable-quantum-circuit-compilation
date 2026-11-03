"""Benchmark I v2 circuit: K=32 dyadic intervals, degree-2 polynomial + Newton-Raphson.

Based on nr_reference_circuit.py; changes:
  - interval selection: one-hot decode of the top 5 bits of x_norm - 8 (the
    interval index i in [0, 23]) via a segment-tree unary iteration (~31 ANDs);
  - coefficient load: XOR table over the 24 one-hot flags (Clifford);
  - Horner with 2 multiply-adds on delta bits of weight 2^-4 .. 2^-db;
  - Newton-Raphson chain and phasing identical to the reference, with the
    widths from the v2 sweep (defaults: the sweep's best config).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from fractions import Fraction

import kickmix as km
import numpy as np

import b1_v2_model as v2m


def _and(builder, inverse, control1, control2, target):
    if inverse:
        builder.del_and(control1, control2, target)
    else:
        builder.init_and(control1, control2, target)


def _add_row(builder, inverse, control, terms, target, scratch, subtract=False):
    sources = km.array([q for q, _ in terms])
    products = km.array([scratch[i] for _, i in terms])
    low = min(i for _, i in terms)
    builder.init_and(control, sources, products)
    if subtract != inverse:
        builder.isub(scratch[low:len(target)], target=target[low:])
    else:
        builder.iadd(scratch[low:len(target)], target=target[low:])
    builder.del_and(control, sources, products)


def _flag_zero_pair(builder, inverse, x, flag):
    top_pair = x[-2:]
    builder.x(top_pair)
    _and(builder, inverse, top_pair[0], top_pair[1], flag)
    builder.x(top_pair)


def _normalize(builder, inverse, x, flags):
    for k in (reversed(range(len(flags))) if inverse else range(len(flags))):
        lanes = (x[2 * k::2], x[2 * k + 1::2])
        if inverse:
            for lane in lanes:
                builder.right_rotate(lane, 1, control=flags[k])
            _flag_zero_pair(builder, inverse, x, flags[k])
        else:
            _flag_zero_pair(builder, inverse, x, flags[k])
            for lane in lanes:
                builder.left_rotate(lane, 1, control=flags[k])


def _select_and_load(builder, prec, x, coefs, anc):
    """QROM-style walk over the top B bits of x_norm (self-inverse).

    At each valid leaf (interval i, address i + base), XORs the interval's
    coefficients into `coefs`. AND ancillas are measured-and-cleaned
    immediately after each subtree (MBU), so `anc` holds B-1 transient qubits
    and nothing persists past this stage; running it again (as the reverse
    stage does) XORs the coefficients back out.
    """
    width = len(x)
    B = (prec.K * 3 // 4 - 1).bit_length()
    num_valid = prec.K * 3 // 4
    base = (1 << B) - num_valid
    bits = [x[width - 1 - k] for k in range(B)]  # bits[0] is the MSB
    table = prec.coefficient_table()

    def load(i, control):
        for j, reg in enumerate(coefs):
            v = int(table[i, j])
            if v:
                builder.gf2_iadd(v, target=reg, control=control)

    def node(control, level, lo, hi):
        if lo >= base + num_valid or hi <= base:
            return
        if hi - lo == 1:
            load(lo - base, control)
            return
        mid = (lo + hi) // 2
        bit, a = bits[level], anc[level - 1]
        builder.x(bit)
        builder.init_and(control, bit, a)  # a = control AND NOT bit
        builder.x(bit)
        node(a, level + 1, lo, mid)
        builder.cx(control, a)             # a = control AND bit
        node(a, level + 1, mid, hi)
        builder.del_and(control, bit, a)   # MBU cleanup

    half = 1 << (B - 1)
    builder.x(bits[0])
    node(bits[0], 1, 0, half)
    builder.x(bits[0])
    node(bits[0], 1, half, 1 << B)


def _load_v2(builder, inverse, prec, onehot, coefs):
    """XOR the coefficients of interval i into coefs (Clifford, self-inverse)."""
    table = prec.coefficient_table()
    for j, reg in enumerate(coefs):
        for i in range(len(onehot)):
            v = int(table[i, j])
            if v:
                builder.gf2_iadd(v, target=reg, control=onehot[i])


def _horner_v2(builder, inverse, prec, x, coefs, scratch):
    """r = c1 - T(c2 delta); y0 = c0 - T(r delta)."""
    width = len(x)
    wc = len(coefs[0])
    B = (prec.K - 1).bit_length()
    delta = [(x[width - 2 - m], m) for m in range(B - 1, min(prec.delta_bits, width - 2) + 1)]
    c0, c1, c2 = coefs
    steps = ((c1, c2), (c0, c1))
    for acc, mult in (reversed(steps) if inverse else steps):
        for control, m in delta:
            _add_row(builder, inverse, control, [(mult[i], i - m) for i in range(m, wc)],
                     acc, scratch, subtract=True)


def _square(builder, inverse, y0, square, scratch):
    wc, wsq = len(y0), len(square)
    y = {k: y0[wc - k] for k in range(1, wc + 1)}
    diagonal = [k for k in range(1, wc + 1) if 2 * k <= wsq]
    controls = km.array([y[k] for k in diagonal])
    targets = km.array([square[wsq - 2 * k] for k in diagonal])
    if not inverse:
        builder.cx(controls, targets)
    for k in range(1, wc + 1):
        terms = [(y[l], wsq - (k + l - 1)) for l in range(k + 1, wc + 1) if k + l - 1 <= wsq]
        if terms:
            _add_row(builder, inverse, y[k], terms, square, scratch)
    if inverse:
        builder.cx(controls, targets)


def _multiply_m(builder, inverse, x, square, z, scratch):
    width, wsq, wz = len(x), len(square), len(z) - 2
    for i in range(width):
        e = wz - wsq + i - width + 2
        _add_row(builder, inverse, x[i],
                 [(square[j], j + e) for j in range(max(0, -e), wsq)], z, scratch)


def _correct(builder, inverse, z):
    constant = (3 << (len(z) - 2)) + 1
    if inverse:
        builder.isub(constant, target=z)
        builder.x(z)
    else:
        builder.x(z)
        builder.iadd(constant, target=z)


def _multiply_y0(builder, inverse, y0, z, p, scratch):
    wc, wz, wp = len(y0), len(z) - 2, len(p) - 1
    for k in range(1, wc + 1):
        e = wp - wz - k
        _add_row(builder, inverse, y0[wc - k],
                 [(z[j], j + e) for j in range(max(0, -e), wz + 2)], p, scratch)


def _denormalize(builder, inverse, flags, y1):
    wp = len(y1) - len(flags) - 1
    for k in (reversed(range(len(flags))) if inverse else range(len(flags))):
        lane = y1[k:k + wp + 2]
        if inverse:
            builder.right_rotate(lane, 1, control=flags[k])
        else:
            builder.left_rotate(lane, 1, control=flags[k])


def _compute_y1(builder, inverse, prec, x, flags, anc, coefs, square, z, y1, scratch):
    y0, p = coefs[0], y1[:len(y1) - len(flags)]
    if inverse:
        with builder.mark('denormalize'):
            _denormalize(builder, inverse, flags, y1)
        with builder.mark('multiply_y0'):
            _multiply_y0(builder, inverse, y0, z, p, scratch)
        with builder.mark('correct'):
            _correct(builder, inverse, z)
        with builder.mark('multiply_m'):
            _multiply_m(builder, inverse, x, square, z, scratch)
        with builder.mark('square'):
            _square(builder, inverse, y0, square, scratch)
        with builder.mark('horner'):
            _horner_v2(builder, inverse, prec, x, coefs, scratch)
        with builder.mark('load_coefficients'):
            _select_and_load(builder, prec, x, coefs, anc)
        with builder.mark('normalize'):
            _normalize(builder, inverse, x, flags)
    else:
        with builder.mark('normalize'):
            _normalize(builder, inverse, x, flags)
        with builder.mark('load_coefficients'):
            _select_and_load(builder, prec, x, coefs, anc)
        with builder.mark('horner'):
            _horner_v2(builder, inverse, prec, x, coefs, scratch)
        with builder.mark('square'):
            _square(builder, inverse, y0, square, scratch)
        with builder.mark('multiply_m'):
            _multiply_m(builder, inverse, x, square, z, scratch)
        with builder.mark('correct'):
            _correct(builder, inverse, z)
        with builder.mark('multiply_y0'):
            _multiply_y0(builder, inverse, y0, z, p, scratch)
        with builder.mark('denormalize'):
            _denormalize(builder, inverse, flags, y1)


def _phase_by_bits(builder, y):
    f = len(y)
    with builder.mark('phase_by_bits'):
        for j in range(f):
            builder.z_pow(y[j], Fraction(2) ** (j - f + 1))


def build_v2_circuit(n_p: int, prec: v2m.V2Precision) -> km.Circuit:
    width = 2 * n_p + 2
    out_bits = prec.output_bits(n_p)
    K, D = prec.K, prec.degree
    assert D == 2 and K == 32, "this builder implements K=32 / D=2"
    builder = km.CircuitBuilder()

    def alloc(length):
        return builder.alloc_qubits(length)

    x = builder.create_quantum_register(width, name='x')
    y1, flags = alloc(out_bits), alloc(n_p)
    anc = alloc((prec.K * 3 // 4 - 1).bit_length() - 1)
    coefs = [alloc(prec.coef_bits) for _ in range(D + 1)]
    square, z = alloc(prec.square_bits), alloc(prec.product_bits + 2)
    scratch = alloc(max(prec.square_bits, prec.product_bits + 2, prec.result_bits + 1))
    _compute_y1(builder, False, prec, x, flags, anc, coefs, square, z, y1, scratch)
    _phase_by_bits(builder, y1)
    _compute_y1(builder, True, prec, x, flags, anc, coefs, square, z, y1, scratch)
    for reg in (y1, flags, anc, *coefs, square, z, scratch):
        builder.free(reg)
    return builder.finish_circuit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Path of the .kmb file to write.')
    parser.add_argument('--wc', type=int, default=17)
    parser.add_argument('--db', type=int, default=12)
    parser.add_argument('--wsq', type=int, default=28)
    parser.add_argument('--wz', type=int, default=28)
    parser.add_argument('--wp', type=int, default=28)
    args = parser.parse_args()
    prec = v2m.V2Precision(K=32, degree=2, coef_bits=args.wc, delta_bits=args.db,
                           square_bits=args.wsq, product_bits=args.wz,
                           result_bits=args.wp)
    circuit = build_v2_circuit(10, prec)
    circuit.to_file(args.output, format='kmb')
    stats = circuit.operation_statistics()
    print(f'Wrote {args.output}: {stats.num_toffoli_like:,} Toffolis, '
          f'{circuit.num_qubits} qubits, {len(circuit):,} ops')


if __name__ == '__main__':
    main()
