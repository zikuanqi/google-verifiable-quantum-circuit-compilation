#!/bin/bash
# b2: find the dirty ancillas and the phase source in the BM-only circuit.
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
for r in range(n, 0, -1):
    B.bm_round_rev(builder, bm, gf, syndromes, r, K_r[r - 1])
builder.free(bm.stack_d); builder.free(bm.stack_r); builder.free(bm.d)
for w in bm.tau: builder.free(w)
for w in bm.sigma: builder.free(w)
circuit = builder.finish_circuit()

sim = km.Simulator(batch_size=8, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_names = [f'S{i}' for i in range(n)]
s_arr = km.array([q for name in s_names for q in circuit.register_data[name]])
synd = np.random.default_rng(1).integers(0, 1 << degree, size=(8, n), dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 8)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
anc = km.array([km.q(k) for k in range(circuit.num_qubits) if km.q(k) not in set(s_arr)])
junk = sim.read_across_shots(anc)
for i in range(len(anc)):
    if junk[i].any():
        shots = np.flatnonzero(junk[i])
        print(f'dirty ancilla q{anc[i]} in shots {shots.tolist()}')
print('phase flips:', [sim.read_shot_phase(i) for i in range(8)])
# which register holds q?
for name, qs in circuit.register_data.items():
    pass
# map qubit -> alloc'd role: print the first few alloc'd qubit indices' users
# sigma0 = ...: just print the bm registers' ids
print('sigma[0] ids:', list(bm.sigma[0]))
print('tau[0] ids:', list(bm.tau[0]))
print('d ids:', list(bm.d))
print('stack_r ids:', list(bm.stack_r))
print('stack_d ids:', list(bm.stack_d))
EOF
