# -*- coding: utf-8 -*-
"""Benchmark III cost analysis: heuristic + analytic ANGLES-QROM costs vs (R,B,C,brot,J).

Angles costs are analytic estimates (uniform QROM per batch, the same structure
as the baseline example circuits); the real number comes from the simulator.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra.costs_prepare import SpinFreeData_V1, QR  # noqa: E402

N = 76
CHI = 10


def angles_qrom_toffoli(rows: int, brot: int, J: int) -> float:
    """Analytic per-pass Toffoli of K=ceil(75/J) uniform QROMs over `rows` entries."""
    K = math.ceil((N - 1) / J)
    total = 0.0
    for k in range(K):
        m1 = J * brot  # bits loaded by batch k
        _, cost = QR(rows, m1)
        total += cost
    return total  # one pass


def model(R: int, B: int, C: int, brot: int, J: int):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=CHI, beta=brot)
    outer_t = m.outer_state_prep()
    inner_t = m.inner_state_prep()
    givens_t = m.givens_rotations()
    outer_q = m.outer_state_prep_qubit_cost()
    inner_q = m.inner_state_prep_qubit_cost()
    rows = N + R * B
    angles_pass = angles_qrom_toffoli(rows, brot, J)
    angles_t = 2 * angles_pass
    # ANGLE_k circuit qubits: selection + J*brot + QROM temp (analytic upper estimate)
    sel = math.ceil(math.log2(rows))
    k_opt = QR(rows, J * brot)[0]
    angles_q = sel + J * brot + math.ceil(math.log2(rows / 2**k_opt)) + (2**k_opt) * J * brot
    heuristic_t = outer_t + inner_t + givens_t
    t_total = angles_t + heuristic_t
    q_total = (outer_q["persistent"] + inner_q["persistent"]) + J * brot \
        + max(outer_q["temporary"], inner_q["temporary"], angles_q - sel - J * brot)
    return {
        "rows": rows, "sel": sel,
        "outer_t": outer_t, "inner_t": inner_t, "givens_t": givens_t,
        "angles_t": angles_t, "heuristic_t": heuristic_t, "t_total": t_total,
        "angles_q": angles_q, "q_total": q_total,
        "angles_within_cap": angles_q <= 128,
    }


def main():
    print(f"{'R':>3} {'B':>3} {'C':>3} {'brot':>4} {'J':>2} | {'rows':>5} {'anglesT':>8} {'heurT':>6} "
          f"{'Ttot':>8} {'anglesQ':>8} {'Qtot':>5} | {'score@49.3':>10}")
    configs = [
        (65, 60, 60, 12, 8),   # baseline
        (40, 60, 60, 12, 8),
        (32, 40, 40, 12, 8),
        (32, 40, 40, 10, 8),
        (65, 60, 60, 12, 16),
        (40, 40, 40, 12, 12),
        (24, 32, 32, 12, 8),
        (16, 24, 24, 12, 8),
        (24, 32, 32, 10, 8),
        (32, 40, 40, 12, 4),
    ]
    for R, B, C, brot, J in configs:
        r = model(R, B, C, brot, J)
        score = 49.301369 * r["t_total"] * r["q_total"]
        print(f"{R:>3} {B:>3} {C:>3} {brot:>4} {J:>2} | {r['rows']:>5} {r['angles_t']:>8.0f} "
              f"{r['heuristic_t']:>6} {r['t_total']:>8.0f} {r['angles_q']:>8.0f} {r['q_total']:>5} "
              f"| {score:>10.3e}  cap={r['angles_within_cap']}")


if __name__ == "__main__":
    main()
