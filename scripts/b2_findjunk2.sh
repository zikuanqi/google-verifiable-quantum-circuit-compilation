#!/bin/bash
# b2: junk scan of the forward-only circuit (find the dirty temp).
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
circuit = builder.finish_circuit()
sim = km.Simulator(batch_size=8, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_arr = km.array([q for name in [f'S{i}' for i in range(n)] for q in circuit.register_data[name]])
synd = np.random.default_rng(1).integers(0, 1 << degree, size=(8, n), dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 8)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
# scan all qubits: which are dirty in any shot, excluding the S-regs and the
# long-lived bm regs (sigma/tau/d/stack)
long = set()
for w in bm.sigma + bm.tau:
    long.update(w)
long.update(bm.d)
long.update(bm.stack_r)
long.update(bm.stack_d)
s_set = set(s_arr)
anc = km.array([km.q(k) for k in range(circuit.num_qubits)
                if km.q(k) not in s_set and km.q(k) not in long])
junk = sim.read_across_shots(anc)
for i, q in enumerate(anc):
    if junk[i].any():
        shots = np.flatnonzero(junk[i]).tolist()
        print(f'dirty temp q{q} in shots {shots}')
EOF
