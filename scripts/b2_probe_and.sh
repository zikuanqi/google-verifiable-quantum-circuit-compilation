#!/bin/bash
# b2: probe init_and / del_and / flip_if_equal semantics (distinct qubits).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np
import kickmix as km

def run(build, nbits, shots, batch=8):
    b = km.CircuitBuilder()
    qs = [b.alloc_qubits(1)[0] for _ in range(nbits)]
    tgt = build(b, qs)
    c = b.finish_circuit()
    circuit = c
    sim = km.Simulator(batch_size=batch, debug_print=False,
                       counting_operations=False, seed=7)
    sim.use_same_registers_as(c)
    arr = km.array([km.q(k) for k in range(circuit.num_qubits)])
    bits = np.zeros((circuit.num_qubits, batch), dtype=bool)
    bits[:shots.shape[0], :] = shots
    sim.write_across_shots(arr, bits)
    sim.do(c)
    out = sim.read_across_shots(arr)
    ph = [sim.read_shot_phase(i) for i in range(batch)]
    return out[tgt].tolist(), ph

shots3 = np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1]], dtype=bool)

# 1. t = a AND c ; del_and ; check t returns 0 (a,c distinct)
def b_and(b, qs):
    a, c = qs[0], qs[1]
    t = b.alloc_qubits(1)[0]
    b.init_and(a, c, t)
    b.del_and(a, c, t)
    return 2
out, ph = run(b_and, 2, shots3)
print('and+del roundtrip t:', out, 'ph:', ph)

# 2. init_and once: t should equal a AND c
def b_and2(b, qs):
    a, c = qs[0], qs[1]
    t = b.alloc_qubits(1)[0]
    b.init_and(a, c, t)
    return 2
out, ph = run(b_and2, 2, shots3)
print('and once t (want a&c):', out, 'ph:', ph)

# 3. flip_if_equal(x2, 0, t) twice same input -> t==0, no phase
def b_flip(b, qs):
    x2 = km.array(qs)
    t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x2, 0, target=t)
    b.flip_if_equal(x2, 0, target=t)
    return 2
out, ph = run(b_flip, 2, shots3)
print('flip double same input t:', out, 'ph:', ph)

# 4. flip twice, x flipped in between
def b_flip2(b, qs):
    x2 = km.array(qs)
    t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x2, 0, target=t)
    b.x(qs[0])
    b.flip_if_equal(x2, 0, target=t)
    return 2
out, ph = run(b_flip2, 2, shots3)
print('flip double changed input t:', out, 'ph:', ph)

# 5. flip once: t should be (x==0)
def b_flip3(b, qs):
    x2 = km.array(qs)
    t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x2, 0, target=t)
    return 2
out, ph = run(b_flip3, 2, shots3)
print('flip once t (want x==0):', out, 'ph:', ph)
EOF