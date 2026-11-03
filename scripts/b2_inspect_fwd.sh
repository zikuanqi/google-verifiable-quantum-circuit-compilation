#!/bin/bash
# b2: inspect the reverse detection inputs (shot 0, k=1).
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

def rv(qs, shot):
    return sum(1 << k for k, q in enumerate(qs) if sim.read_within_shot(q, shot))

builder = km.CircuitBuilder()
syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
bm = B.BMRegs(builder, n, degree)
K_r = [min(r, n // 2) for r in range(1, n + 1)]
for r in range(1, n + 1):
    B.bm_round_fwd(builder, bm, gf, syndromes, r, K_r[r - 1])
circuit = builder.finish_circuit()
sim = km.Simulator(batch_size=1, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_arr = km.array([q for name in [f'S{i}' for i in range(n)] for q in circuit.register_data[name]])
synd = np.array([[7, 8, 12, 15, 0, 2]], dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 1)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
print('post-forward:')
for w in range(n // 2 + 1):
    print(f'  sigma[{w}]={rv(list(bm.sigma[w]), 0)} tau[{w}]={rv(list(bm.tau[w]), 0)}')
d = rv(list(bm.d), 0); d = d - 256 if d >= 128 else d
print('  d =', d)
EOF
