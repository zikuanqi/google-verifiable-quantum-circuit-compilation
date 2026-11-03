#!/bin/bash
# b2: bisect the stage-1 circuit (which stage breaks the phases).
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

def run_check(build, label):
    circuit = build()
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
    ph = [sim.read_shot_phase(i) for i in range(batch)]
    print(f'{label}: restore={ok} junk={bool(junk.any())} phases={ph}')

def build_bm_only():
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    bm = B.BMRegs(builder, n, degree)
    K_r = [min(r, n // 2) for r in range(1, n + 1)]
    for r in range(1, n + 1):
        B.bm_round_fwd(builder, bm, gf, syndromes, r, K_r[r - 1])
    for r in range(n, 0, -1):
        B.bm_round_rev(builder, bm, gf, syndromes, r, K_r[r - 1])
    for s in bm.stk:
        builder.free(s)
    builder.free(bm.d)
    for w in bm.tau:
        builder.free(w)
    for w in bm.sigma:
        builder.free(w)
    return builder.finish_circuit()

def build_bm_norm():
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    bm = B.BMRegs(builder, n, degree)
    keep = builder.alloc_qubits(degree)
    K_r = [min(r, n // 2) for r in range(1, n + 1)]
    for r in range(1, n + 1):
        B.bm_round_fwd(builder, bm, gf, syndromes, r, K_r[r - 1])
    B.normalize_sigma(builder, bm.sigma, keep, gf)
    B.unnormalize_sigma(builder, bm.sigma, keep, gf)
    for r in range(n, 0, -1):
        B.bm_round_rev(builder, bm, gf, syndromes, r, K_r[r - 1])
    builder.free(keep)
    for s in bm.stk:
        builder.free(s)
    builder.free(bm.d)
    for w in bm.tau:
        builder.free(w)
    for w in bm.sigma:
        builder.free(w)
    return builder.finish_circuit()

def build_sqrt_only():
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    sig = [builder.alloc_qubits(degree) for _ in range(n // 2 + 1)]
    B.sqrt_transform(builder, sig, gf, reverse=False)
    B.sqrt_transform(builder, sig, gf, reverse=True)
    for w in sig:
        builder.free(w)
    return builder.finish_circuit()

run_check(build_bm_only, 'BM only')
run_check(build_bm_norm, 'BM+norm')
run_check(build_sqrt_only, 'sqrt only')
EOF
