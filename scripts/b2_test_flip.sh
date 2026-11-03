#!/bin/bash
# b2: flip_if_equal on circuit-evolved vs written values (seed variants).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
import numpy as np
import kickmix as km

def test(evolve, seed):
    b = km.CircuitBuilder()
    a = b.alloc_qubits(4)
    t = b.alloc_qubits(1)[0]
    if evolve:
        # evolve a = 13 via X gates
        for k in range(4):
            if (13 >> k) & 1:
                b.x(a[k])
    b.flip_if_equal(a, 0, target=t)
    out = b.alloc_qubits(1)
    b.cx(t, out[0])
    c = b.finish_circuit()
    sim = km.Simulator(batch_size=1, debug_print=False, counting_operations=False, seed=seed)
    sim.use_same_registers_as(c)
    if not evolve:
        for k in range(4):
            sim.write_within_shot(a[k], 0, (13 >> k) & 1)
    sim.do(c)
    return sim.read_within_shot(out[0], 0), sim.read_shot_phase(0)

for evolve in (False, True):
    for seed in (1, 7):
        v, ph = test(evolve, seed)
        print(f'evolve={evolve} seed={seed}: flag={v} phase={ph}')
EOF
