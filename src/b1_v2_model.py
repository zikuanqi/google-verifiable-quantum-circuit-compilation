"""Benchmark I v2: K-interval degree-D polynomial + Newton-Raphson phasing circuit.

Architecture changes vs the reference (nr_reference_circuit.py):
  - K dyadic intervals selected by the top B=log2(K) bits of x_norm (no
    comparisons), each fitted offline (b1_fit.py) with degree D = 1 or 2
    best-fit polynomials, coefficients quantized to `coef_bits`.
  - Horner evaluation with D multiply-adds instead of 3 (cubic).
  - Coefficients loaded by an XOR table over the B selector bits (Clifford)
    into `coef_bits`-wide registers (D+1 of them).
  - Newton-Raphson chain, register widths, and phasing as in the reference,
    with all widths parameterized.

The fixed-point model `v2_model` mirrors this circuit exactly and is used to
sweep widths on Windows before building anything in kickmix.
"""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np

import b1_fit


@dataclasses.dataclass(frozen=True)
class V2Precision:
    K: int = 16                 # number of dyadic intervals
    degree: int = 2             # polynomial degree per interval (1 or 2)
    coef_bits: int = 17         # coefficient / Horner intermediate width
    delta_bits: int = 14        # bits of delta used by Horner multipliers
    square_bits: int = 28       # y0^2
    product_bits: int = 28      # M y0^2 and 3 - M y0^2 (plus two integer bits)
    result_bits: int = 29       # p = y0 (3 - M y0^2) (plus one integer bit)

    @property
    def log2k(self) -> int:
        return (self.K - 1).bit_length()

    def coefficient_table(self) -> np.ndarray:
        """(K, D+1) integer coefficients, scaled by 2^coef_bits, best-fit per interval."""
        h = 4.0 / self.K
        table = np.zeros((self.K, self.degree + 1), dtype=np.int64)
        for i in range(self.K):
            lo, hi = 1.0 + i * h, 1.0 + (i + 1) * h
            coefs, _ = b1_fit.fit_poly_interval(lo, hi, self.degree)
            table[i] = np.round(coefs * 2**self.coef_bits).astype(np.int64)
        # Keep c0 strictly below 2^coef_bits so the register never overflows at
        # delta = 0 (e.g. x = 1, where 1/sqrt(1) = 1 exactly), and keep the
        # first interval's c0 <= 131064 (the reference's value) so that
        # p = y0(3 - y0^2) stays below its register limit 2^(result_bits+1)
        # at x = 1 (the floor-shifted sum rounds up there).
        table[:, 0] = np.minimum(table[:, 0], 2**self.coef_bits - 1)
        table[0, 0] = min(table[0, 0], 131064)
        return table

    def output_bits(self, n_p: int) -> int:
        return self.result_bits + n_p + 1


def v2_model(n_p: int, prec: V2Precision, xs: np.ndarray) -> np.ndarray:
    """y1 = poly(M) -> Newton-Raphson, for each x, mirroring the v2 circuit.

    Returns y1 with `y1 / 2^output_bits` ~= 1/sqrt(x). Raises OverflowError if
    an intermediate would overflow (the circuit would silently wrap).
    """
    width, K, D, wc, db, wsq, wz, wp = (2 * n_p + 2, prec.K, prec.degree,
                                        prec.coef_bits, prec.delta_bits,
                                        prec.square_bits, prec.product_bits,
                                        prec.result_bits)
    xs = np.asarray(xs, dtype=np.int64)
    if not ((1 <= xs) & (xs < 2**width)).all():
        raise ValueError("x must lie in [1, 2^(width - 1)).")
    B = (K - 1).bit_length()
    table = prec.coefficient_table()

    def bit(v, i):
        return (v >> i) & 1

    def product(rows, operand, out_bits, operand_bits, limit):
        acc = np.zeros_like(xs)
        for ctl, weight in rows:
            e = out_bits - operand_bits + weight
            acc += ctl * (operand << e if e >= 0 else operand >> -e)
        if (acc >= 2**limit).any():
            raise OverflowError("Intermediate register overflow.")
        return acc

    # 1. Normalize (same as the reference).
    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)

    # 2. Interval selection: i = floor((M-1)/2^(2-B)) in [0, K).
    i = (x_norm - (1 << (width - 2))) >> (width - B)
    # delta = (x_norm & ((1 << (width - 2 - B)) - 1)) / 2^(width - 2 - B), in [0, 2^(2-B)).
    mask = (1 << (width - 2 - B)) - 1
    d_int = x_norm & mask  # fixed point with (width - 2 - B) fractional bits
    # delta bits of weight 2^-m: m = B-1, B, ..., db (bit index (width-2-B-m) of x_norm).
    delta_rows = [(bit(x_norm, width - 2 - m), -m) for m in range(B - 1, min(db, width - 2) + 1)]

    # 3. Horner, degree D: r = c_D; for acc in (c_{D-1}, ..., c_0): acc -= T(r*delta); r = acc
    #    y0 ends in c0's register.
    dreg = table[i, D]
    r = dreg
    for j in range(D - 1, -1, -1):
        y0 = r = table[i, j] - product(delta_rows, r, wc, wc, wc)
        if ((y0 <= 0) | (y0 >= 2**wc)).any():
            raise OverflowError("Horner intermediate does not fit its register.")

    # 4. Newton-Raphson (identical structure to the reference).
    y = {k: bit(y0, wc - k) for k in range(1, wc + 1)}
    square = np.zeros_like(xs)
    for k in range(1, wc + 1):
        if 2 * k <= wsq:
            square += y[k] << (wsq - 2 * k)
        for l in range(k + 1, wc + 1):
            if k + l - 1 <= wsq:
                square += (y[k] & y[l]) << (wsq - (k + l - 1))
    m_rows = [(bit(x_norm, i2), i2 - width + 2) for i2 in range(width)]
    z = product(m_rows, square, wz, wsq, wz + 2)
    correction = (3 << wz) - z
    p = product([(y[k], -k) for k in range(1, wc + 1)], correction, wp, wz, wp + 1)
    return p << s


def v2_relative_error(n_p: int, prec: V2Precision, xs: np.ndarray) -> np.ndarray:
    exact = 1 / np.sqrt(np.asarray(xs, dtype=np.float64))
    approx = v2_model(n_p, prec, xs) / 2.0**prec.output_bits(n_p)
    return np.abs(approx - exact) / exact


if __name__ == "__main__":
    # Quick check of the default v2 config over the whole domain.
    n_p = 10
    max_x = 3 * (2**n_p - 2) ** 2
    xs = np.arange(1, max_x + 1, dtype=np.int64)
    for K, D in ((16, 2), (32, 2), (8, 3), (64, 1)):
        prec = V2Precision(K=K, degree=D)
        try:
            rel = v2_relative_error(n_p, prec, xs)
            print(f"K={K} D={D}: worst rel err 2^{math.log2(rel.max()):.2f} at x={rel.argmax()+1}")
        except OverflowError as e:
            print(f"K={K} D={D}: overflow: {e}")
