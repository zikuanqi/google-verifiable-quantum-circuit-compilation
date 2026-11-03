#!/bin/bash
# b2: detection with the real post-forward inputs.
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
import numpy as np
import kickmix as km
import b2_build as B

m, n, degree = 15, 6, 4
gf = km.GF2Field(degree)

builder = km.CircuitBuilder()
syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
bm = B.BMRegs(builder, n, degree)
K_r = [min(r, n // 2) for r in range(1, n + 1)]
for r in range(1, n + 1):
    B.bm_round_fwd(builder, bm, gf, syndromes, r, K_r[r - 1])
# replicate the detection and copy the flags out
t0_nz = builder.alloc_qubits(1)[0]
builder.flip_if_equal(bm.tau[0], 0, target=t0_nz)
dle0 = builder.alloc_qubits(1)[0]
builder.cx(bm.d[7], dle0)
deq0 = builder.alloc_qubits(1)[0]
builder.flip_if_equal(bm.d, 0, target=deq0)
builder.cx(deq0, dle0)
eq_035 = builder.alloc_qubits(1)[0]
builder.flip_if_equal(bm.tau[0], bm.tau[3], target=eq_035)
n0 = B._not(builder, t0_nz)
neq = B._not(builder, eq_035)
rep = B._and(builder, n0, dle0)
rep2 = B._and(builder, rep, neq)
out = builder.alloc_qubits(6)
builder.cx([rep2, rep, n0, neq, dle0, eq_035], out)
circuit = builder.finish_circuit()
sim = km.Simulator(batch_size=1, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_arr = km.array([q for name in [f'S{i}' for i in range(n)] for q in circuit.register_data[name]])
synd = np.array([[7, 8, 12, 15, 0, 2]], dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 1)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
vals = [sim.read_within_shot(out[k], 0) for k in range(6)]
print('rep2, rep, n0, neq, dle0, eq_035 =', vals)
def rv(qs):
    return sum(1 << k for k, q in enumerate(qs) if sim.read_within_shot(q, 0))
print('tau0:', rv(list(bm.tau[0])), 'tau3:', rv(list(bm.tau[3])), 'd:', rv(list(bm.d)))
EOF
