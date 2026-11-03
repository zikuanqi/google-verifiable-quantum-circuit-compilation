# -*- coding: utf-8 -*-
"""Benchmark III calibrated resource-cost model (replicates verify.py's formulas).

Key calibration (from the shipped baseline and the reference circuit builder):
  * ANGLES circuits (unary-iteration segment-tree QROM, `angle_k_reference_circuit.py`)
    cost A-2 Toffoli-like gates per circuit per pass where A = 76 + R*B:
    baseline A=3976 -> 3974/pass/circuit -> 2*10*3974 = 79480 = README value.
    Output bits load through CX (free), so Toffolis do NOT depend on J*brot.
  * ANGLE_k qubits = sel + J*brot + (sel-1)  (sel = ceil(log2(A))), cap 128.
  * Q_total = outer_pers + inner_pers + J*brot + max(outer_temp, inner_temp, sel-1)
  * heuristic = outer + inner + 2*(N-1)*(2*brot-1)  via SpinFreeData_V1.

Real ANGLES Toffolis can only be measured with kickmix (Linux); this model is the
best Windows-side estimate and matches the baseline exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402

N = 76
CHI = 10
QUBIT_CAP = 128


def sel_bits(rows: int) -> int:
    return max(1, (rows - 1).bit_length())


def max_j(rows: int, brot: int) -> int:
    """Largest batch size J with the ANGLE circuit inside 128 qubits."""
    s = sel_bits(rows)
    return max(1, (QUBIT_CAP - 2 * s + 1) // brot)


def angles_toffoli(rows: int, brot: int, J: int) -> int:
    """Toffolis of the ANGLE family (forward+reverse), reference-QROM estimate."""
    K = math.ceil((N - 1) / J)
    return 2 * K * max(0, rows - 2)


def resource(R: int, B: int, C: int, brot: int, J: int | None = None) -> dict:
    """T_total, Q_total and the breakdown for one configuration."""
    rows = N + R * B
    s = sel_bits(rows)
    if J is None:
        J = max_j(rows, brot)
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=CHI, beta=brot)
    outer_t = m.outer_state_prep()
    inner_t = m.inner_state_prep()
    givens_t = m.givens_rotations()
    outer_q = m.outer_state_prep_qubit_cost()
    inner_q = m.inner_state_prep_qubit_cost()
    angles_t = angles_toffoli(rows, brot, J)
    angles_q = s + J * brot + (s - 1)
    heuristic_t = outer_t + inner_t + givens_t
    persistent = outer_q["persistent"] + inner_q["persistent"]
    temporary = max(outer_q["temporary"], inner_q["temporary"], s - 1)
    return {
        "R": R, "B": B, "C": C, "brot": brot, "J": J,
        "rows": rows, "sel": s, "K": math.ceil((N - 1) / J),
        "angles_t": angles_t, "angles_q": angles_q,
        "outer_t": outer_t, "inner_t": inner_t, "givens_t": givens_t,
        "outer_q": outer_q, "inner_q": inner_q,
        "t_total": angles_t + heuristic_t,
        "q_total": persistent + J * brot + temporary,
        "angles_within_cap": angles_q <= QUBIT_CAP,
    }


def report(cfg: dict, lam_eff: float = 49.301369) -> str:
    r = resource(cfg["R"], cfg["B"], cfg["C"], cfg["brot"], cfg.get("J"))
    ok = r["angles_within_cap"]
    return (f"(R,B,C,brot,J)=({r['R']},{r['B']},{r['C']},{r['brot']},{r['J']}) "
            f"rows={r['rows']:5d} | angles {r['angles_t']:7d} Toff {r['angles_q']:4d} q | "
            f"heur {r['outer_t']+r['inner_t']+r['givens_t']:6d} | "
            f"T={r['t_total']:7d} Q={r['q_total']:4d} cap={'OK ' if ok else 'BAD'} | "
            f"score@{lam_eff:.2f} = {lam_eff * r['t_total'] * r['q_total']:.3e}")


def sweep() -> None:
    print(f"{'R':>3} {'B':>3} {'C':>3} {'brot':>4} {'J':>2} {'rows':>5} {'K':>2} | "
          f"{'anglesT':>7} {'heurT':>6} {'Ttot':>7} {'anglesQ':>4} {'Qtot':>5} | {'score@49.3':>10} cap")
    configs = [
        (65, 60, 60, 12, 8),   # baseline
        (65, 60, 60, 14, 7),
        (65, 60, 60, 13, 8),
        (96, 8, 8, 12, None),
        (48, 16, 16, 12, None),
        (32, 24, 24, 12, None),
        (24, 32, 32, 12, None),
        (20, 40, 40, 12, None),
        (32, 32, 32, 12, None),
        (16, 24, 24, 12, None),
        (16, 24, 24, 10, None),
        (16, 24, 24, 14, None),
        (32, 24, 24, 14, None),
        (32, 24, 24, 13, None),
        (48, 16, 16, 14, None),
        (48, 16, 16, 10, None),
        (96, 8, 8, 14, None),
        (24, 32, 32, 14, None),
        (12, 32, 32, 12, None),
        (10, 40, 40, 12, None),
        (12, 24, 24, 12, None),
    ]
    for R, B, C, brot, J in configs:
        r = resource(R, B, C, brot, J)
        print(f"{R:>3} {B:>3} {C:>3} {brot:>4} {r['J']:>2} {r['rows']:>5} {r['K']:>2} | "
              f"{r['angles_t']:>7} {r['outer_t']+r['inner_t']+r['givens_t']:>6} {r['t_total']:>7} "
              f"{r['angles_q']:>4} {r['q_total']:>5} | "
              f"{49.301369 * r['t_total'] * r['q_total']:>10.3e} "
              f"{'OK' if r['angles_within_cap'] else 'BAD'}")


if __name__ == "__main__":
    sweep()
