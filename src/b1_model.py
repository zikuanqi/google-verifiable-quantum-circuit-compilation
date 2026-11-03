"""Benchmark I: kickmix-free fixed-point model (extracted from the official
nr_reference_circuit.py so it can run on plain numpy / Windows).

`fixed_point_model(n_p, prec, xs)` reproduces exactly the y1 value the
reference circuit computes for each x; `relative_error` reports the relative
error vs 1/sqrt(x).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

# Cubic coefficients of the paper for 1/sqrt(x) on [1, 1.5] (`A`) and
# [1.5, 2] (`B`), with alternating signs supplied by the circuit.
POLY_A = (0.99994132489119882162, 0.49609891915903542303, 0.33261112772430493331, 0.14876762006038398086)
POLY_B = (0.81648515205385221995, 0.27136515484240234115, 0.12756148214815175348, 0.044753028579153842218)


@dataclasses.dataclass(frozen=True)
class Precision:
    """Fractional bit widths of the pipeline's registers (baseline tuned for n_p = 10)."""
    delta_bits: int = 14  # Bits of delta used by the Horner multipliers.
    coef_bits: int = 17  # Coefficients and Horner intermediates r1, r2, y0.
    square_bits: int = 28  # y0^2.
    product_bits: int = 28  # M y0^2 and 3 - M y0^2 (plus two integer bits).
    result_bits: int = 29  # p = y0 (3 - M y0^2) (plus one integer bit).

    def coefficient_table(self) -> list[list[int]]:
        """`table[case][j]` = coefficient magnitude `j` for interval `case`, times `2^coef_bits`."""
        cases = [POLY_A, POLY_B,
                 [a / (2**j * math.sqrt(2)) for j, a in enumerate(POLY_A)],
                 [b / (2**j * math.sqrt(2)) for j, b in enumerate(POLY_B)]]
        return [[round(c * 2**self.coef_bits) for c in case] for case in cases]

    def output_bits(self, n_p: int) -> int:
        return self.result_bits + n_p + 1


def fixed_point_model(n_p: int, prec: Precision, xs: np.ndarray) -> np.ndarray:
    """`y1` as computed by `build_nr_circuit` for each `x` in `xs`.

    `y1 / 2^prec.output_bits(n_p)` approximates `1 / sqrt(x)`. Raises if an
    intermediate would overflow its register, which the circuit would silently
    wrap.
    """
    width, d, wc, wsq, wz, wp = 2 * n_p + 2, *dataclasses.astuple(prec)
    xs = np.asarray(xs, dtype=np.int64)
    if not ((1 <= xs) & (xs < 2**width)).all():
        raise ValueError("x must lie in [1, 2^(width - 1)).")

    def bit(v: np.ndarray, i: int) -> np.ndarray:
        return (v >> i) & 1

    def product(rows: list[tuple[np.ndarray, int]], operand: np.ndarray, out_bits: int,
                operand_bits: int, limit: int) -> np.ndarray:
        """`T(sum_rows ctl 2^weight operand)` in units of `2^-out_bits`, checked `< 2^limit`."""
        acc = np.zeros_like(xs)
        for ctl, weight in rows:
            e = out_bits - operand_bits + weight
            acc += ctl * (operand << e if e >= 0 else operand >> -e)
        if (acc >= 2**limit).any():
            raise OverflowError("Intermediate register overflow.")
        return acc

    # 1. Normalize: s leading zero pairs, M = x_norm / 2^(width - 2) in [1, 4).
    s = (width - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    x_norm = xs << (2 * s)
    top, second, third = bit(x_norm, width - 1), bit(x_norm, width - 2), bit(x_norm, width - 3)

    # 2. Initial guess: coefficients by case, delta bits of weight 2^-m for m = 1..d.
    table = np.array(prec.coefficient_table(), dtype=np.int64)
    c0, c1, c2, c3 = (table[np.where(top == 0, third, 2 + second), j] for j in range(4))
    delta = [(top & third, -1)] + [(bit(x_norm, width - 2 - m), -m) for m in range(2, min(d, width - 2) + 1)]
    y0, mult = c0, c3
    for acc in (c2, c1, c0):  # r1, r2, y0.
        y0 = mult = acc - product(delta, mult, wc, wc, wc)
        if ((y0 <= 0) | (y0 >= 2**wc)).any():
            raise OverflowError("Horner intermediate does not fit its register.")

    # 3. Newton-Raphson: y0^2 as diagonal bits plus doubled cross terms.
    y = {k: bit(y0, wc - k) for k in range(1, wc + 1)}  # Bit of weight 2^-k.
    square = np.zeros_like(xs)
    for k in range(1, wc + 1):
        if 2 * k <= wsq:
            square += y[k] << (wsq - 2 * k)
        for l in range(k + 1, wc + 1):
            if k + l - 1 <= wsq:
                square += (y[k] & y[l]) << (wsq - (k + l - 1))
    m_rows = [(bit(x_norm, i), i - width + 2) for i in range(width)]
    z = product(m_rows, square, wz, wsq, wz + 2)
    correction = (3 << wz) - z
    p = product([(y[k], -k) for k in range(1, wc + 1)], correction, wp, wz, wp + 1)
    return p << s


def relative_error(n_p: int, prec: Precision, xs: np.ndarray) -> np.ndarray:
    """`|y1 / 2^F - 1 / sqrt(x)| sqrt(x)` of the model for each `x` in `xs`."""
    exact = 1 / np.sqrt(np.asarray(xs, dtype=np.float64))
    approx = fixed_point_model(n_p, prec, xs) / 2.0**prec.output_bits(n_p)
    return np.abs(approx - exact) / exact
