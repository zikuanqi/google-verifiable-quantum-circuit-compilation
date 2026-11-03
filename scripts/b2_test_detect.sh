#!/bin/bash
# b2: isolated test of the reverse detection pieces.
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
import numpy as np
import kickmix as km
import b2_build as B

degree = 4
b = km.CircuitBuilder()
tau0 = b.alloc_qubits(degree)
tauL = b.alloc_qubits(degree)
d = b.alloc_qubits(8)
# detection
t0_nz = b.alloc_qubits(1)[0]
b.flip_if_equal(tau0, 0, target=t0_nz)
dle0 = b.alloc_qubits(1)[0]
b.cx(d[7], dle0)
deq0 = b.alloc_qubits(1)[0]
b.flip_if_equal(d, 0, target=deq0)
b.cx(deq0, dle0)
eq_035 = b.alloc_qubits(1)[0]
b.flip_if_equal(tau0, tauL, target=eq_035)
n0 = B._not(b, t0_nz)
neq = B._not(b, eq_035)
rep = B._and(b, n0, dle0)
rep2 = B._and(b, rep, neq)
# keep the results for reading: copy rep2 etc into fresh kept qubits
out = b.alloc_qubits(4)
b.cx([rep2, rep, n0, dle0], out)
c = b.finish_circuit()
sim = km.Simulator(batch_size=8, debug_print=False, counting_operations=False, seed=1)
sim.use_same_registers_as(c)
# write tau0 = 13, tauL = 1, d = 0
for shot in range(8):
    for k in range(4):
        sim.write_within_shot(tau0[k], shot, (13 >> k) & 1)
        sim.write_within_shot(tauL[k], shot, (1 >> k) & 1)
    for k in range(8):
        sim.write_within_shot(d[k], shot, 0)
sim.do(c)
for shot in range(4):
    vals = [sim.read_within_shot(out[k], shot) for k in range(4)]
    print(f'shot {shot}: rep2={vals[0]} rep={vals[1]} n0={vals[2]} dle0={vals[3]}')
EOF
