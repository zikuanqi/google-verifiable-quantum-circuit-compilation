#!/bin/bash
# b2: quick reversibility test of the BM+normalize+product+sqrt stages (small).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
import numpy as np
import kickmix as km
import b2_build

m, n, degree = 15, 6, 4
gf = km.GF2Field(degree)
circuit = b2_build.build_circuit(m=m, n=n, degree=degree)
st = circuit.operation_statistics()
print(f'circuit: {st.num_toffoli_like} Toffolis, {circuit.num_qubits} qubits, {len(circuit)} ops')

sim = km.Simulator(batch_size=8, debug_print=False, counting_operations=False, seed=7)
sim.use_same_registers_as(circuit)
s_names = [f'S{i}' for i in range(n)]
s_arr = km.array([q for name in s_names for q in circuit.register_data[name]])
batch = 8
synd = np.random.default_rng(1).integers(0, 1 << degree, size=(batch, n), dtype=np.int64)
bits = ((synd[:, :, None] >> np.arange(degree)) & 1 == 1).transpose(1, 2, 0).reshape(-1, batch)
sim.write_across_shots(s_arr, bits)
sim.do(circuit)
out = sim.read_across_shots(s_arr)
ok = (out == bits).all()
anc = km.array([km.q(k) for k in range(circuit.num_qubits) if km.q(k) not in set(s_arr)])
junk = sim.read_across_shots(anc) if len(anc) else np.zeros((0, batch), dtype=bool)
print('syndrome restore ok:', ok)
print('ancilla junk any:', bool(junk.any()))
print('phase flips:', [sim.read_shot_phase(i) for i in range(batch)])
EOF
