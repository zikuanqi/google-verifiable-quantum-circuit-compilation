#!/bin/bash
# b2: probe ixor_gf2_mul phases on basis states: single mul, double mul,
# gated mul, and the mul+HMR+correction protocol.
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np
import kickmix as km

gf = km.GF2Field(4)

def run(build, nq, shots, batch=None):
    b = km.CircuitBuilder()
    qs = [b.alloc_qubits(4) for _ in range(nq)]
    tgt = build(b, qs, gf)
    c = b.finish_circuit()
    batch = batch or shots.shape[1]
    sim = km.Simulator(batch_size=batch, debug_print=False,
                       counting_operations=False, seed=7)
    sim.use_same_registers_as(c)
    arr = km.array([km.q(k) for k in range(c.num_qubits)])
    bits = np.zeros((c.num_qubits, batch), dtype=bool)
    for q in range(min(shots.shape[0], c.num_qubits)):
        bits[4*q:4*q+4, :] = shots[q]
    sim.write_across_shots(arr, bits)
    sim.do(c)
    out = sim.read_across_shots(arr)
    ph = [sim.read_shot_phase(i) for i in range(batch)]
    return out, ph

shots = np.array([
    [0b0011, 0b0011, 0b1010, 0b1010],
    [0b0101, 0b0101, 0b0111, 0b0111],
    [0b0111, 0b0111, 0b1001, 0b1001],
    [0b1011, 0b1011, 0b1101, 0b1101],
], dtype=np.int64)

# 1. single mul: t = a*b ; phase?
def b_mul(b, qs, gf):
    a, t = qs[0], qs[1]
    b.ixor_gf2_mul(a, t, target=t, field=gf)
    return 1
out, ph = run(b_mul, 2, shots)
print('single mul t=', [hex(int(x)) for x in out[1]], 'ph:', ph)

# 2. double mul (clear): phase back to 0?
def b_mul2(b, qs, gf):
    a, t = qs[0], qs[1]
    b.ixor_gf2_mul(a, t, target=t, field=gf)
    b.ixor_gf2_mul(a, t, target=t, field=gf)
    return 1
out, ph = run(b_mul2, 2, shots)
print('double mul t=', [hex(int(x)) for x in out[1]], 'ph:', ph)

# 3. gated mul with control=1 (written), double
def b_mul3(b, qs, gf):
    a, t, ctl = qs[0], qs[1], qs[2]
    b.x(ctl[0])
    b.ixor_gf2_mul(a, t, target=t, field=gf, control=ctl[0])
    b.ixor_gf2_mul(a, t, target=t, field=gf, control=ctl[0])
    b.x(ctl[0])
    return 1
out, ph = run(b_mul3, 3, shots)
print('gated double mul t=', [hex(int(x)) for x in out[1]], 'ph:', ph)

# 4. mul with target starting nonzero, forward+reverse operands swapped values:
#    fwd: t0 ^= a0*b0 ; rev: t1 ^= a1*b1 (different operands) -- phase residue?
def b_mul4(b, qs, gf):
    a, t = qs[0], qs[1]
    b.ixor_gf2_mul(a, t, target=t, field=gf)     # t = a*old_t ... target doubles as operand
    return 1
out, ph = run(b_mul4, 2, shots)
print('mul t(operand=target) t=', [hex(int(x)) for x in out[1]], 'ph:', ph)

# 5. HMR protocol: t = mul(a,b); hmr(t -> c); re-mul into fresh t2; apply
#    neg correction <t2, c> via phase_by_product?  check phase.
def b_mul5(b, qs, gf):
    a, bb = qs[0], qs[1]
    t = b.alloc_qubits(4)
    b.init_gf2_mul(a, bb, target=t, field=gf)
    cb = b.alloc_bits(4)
    b.hmr(t, cb)
    b.free(t)
    t2 = b.alloc_qubits(4)
    b.init_gf2_mul(a, bb, target=t2, field=gf)
    b.gf2_phase_by_product(a, bb, mask=cb, field=gf)
    b.del_gf2_mul(a, bb, target=t2, field=gf)
    b.free(t2)
    return 0
out, ph = run(b_mul5, 2, shots)
print('HMR+phase_by_product: ph:', ph)
EOF