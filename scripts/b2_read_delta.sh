#!/bin/bash
# b2: read the delta_t temp value after one round.
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
sim = km.Simulator(batch_size=1, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_arr = km.array([q for name in [f'S{i}' for i in range(n)] for q in circuit.register_data[name]])
synd = np.array([[7, 8, 12, 15, 0, 2]], dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, 1)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
def rv(qs):
    return sum(1 << k for k, q in enumerate(qs) if sim.read_within_shot(q, 0))
print('delta_t (q75-78):', rv(km.array([km.q(k) for k in range(75, 79)])))
print('sigma0:', rv(list(bm.sigma[0])), 'tau0:', rv(list(bm.tau[0])))
print('phase:', sim.read_shot_phase(0))
EOF
