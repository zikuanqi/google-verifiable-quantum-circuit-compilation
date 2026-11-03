#!/bin/bash
# b2: find which temp is dirty after ONE forward round.
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
B.bm_round_fwd(builder, bm, gf, syndromes, 1, 1)
circuit = builder.finish_circuit()
print('num_qubits:', circuit.num_qubits)
sim = km.Simulator(batch_size=8, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_arr = km.array([q for name in [f'S{i}' for i in range(n)] for q in circuit.register_data[name]])
synd = np.random.default_rng(1).integers(0, 1 << degree, size=(8, n), dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 8)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
long = set()
for w in bm.sigma + bm.tau:
    long.update(w)
long.update(bm.d); long.update(bm.stack_r); long.update(bm.stack_d)
s_set = set(s_arr)
anc = km.array([km.q(k) for k in range(circuit.num_qubits)
                if km.q(k) not in s_set and km.q(k) not in long])
junk = sim.read_across_shots(anc)
for i, q in enumerate(anc):
    if junk[i].any():
        print(f'dirty q{q} in shots {np.flatnonzero(junk[i]).tolist()}')
print('num dirty:', int(junk.any(axis=1).sum()))
print('phase flips:', [sim.read_shot_phase(i) for i in range(8)])
EOF
