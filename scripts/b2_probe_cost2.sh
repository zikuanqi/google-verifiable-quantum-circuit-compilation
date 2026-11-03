#!/bin/bash
# b2: follow-up probe: num_qubits semantics, index reuse, phase_by_product cost.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
$PY - <<'EOF'
import kickmix as km
gf = km.GF2Field(12)

# num_qubits with gates
b = km.CircuitBuilder()
q1 = b.alloc_qubits(100)
b.x(q1[99])
b.free(q1)
q2 = b.alloc_qubits(50)
b.x(q2[49])
c = b.finish_circuit()
print('num_qubits (gate on idx99, free, gate on realloc idx49):', c.num_qubits)

b = km.CircuitBuilder()
q = b.alloc_qubits(5)
b.x(q[4]); b.free(q); q = b.alloc_qubits(3); b.x(q[2])
c = b.finish_circuit()
print('num_qubits (idx4, free, realloc idx0..2):', c.num_qubits)

# phase_by_product cost + mask semantics
b = km.CircuitBuilder()
lhs = b.alloc_qubits(12); rhs = b.alloc_qubits(12)
mask = b.alloc_bits(12)
for i in range(12):
    b.bit_store(mask[i], True)
b.gf2_phase_by_product(lhs, rhs, mask=mask, field=gf)
c = b.finish_circuit()
st = c.operation_statistics()
print('phase_by_product toffoli:', st.num_toffoli_like, 'ops:', st.num_instructions)

# flip_if_equal quantum vs quantum 12-bit
b = km.CircuitBuilder()
a = b.alloc_qubits(12); bb = b.alloc_qubits(12); t = b.alloc_qubits(1)[0]
b.flip_if_equal(a, bb, target=t)
b.flip_if_equal(a, bb, target=t)
c = b.finish_circuit()
print('flip_if_equal(q,q,12) x2 toffoli:', c.operation_statistics().num_toffoli_like)

# simulate a full round skeleton to validate register-level semantics:
# rotate by 1 on 36-qubit lane via left_rotate, plus cswap swap of two words
b = km.CircuitBuilder()
sigma = [b.alloc_qubits(12) for _ in range(3)]
tau = [b.alloc_qubits(12) for _ in range(3)]
for w in range(3):
    lane = km.array([r[w] for r in tau])
    b.left_rotate(lane, 1)
flag = b.alloc_qubits(1)[0]
for i in range(3):
    b.cswap(flag, sigma[i], tau[i])
c = b.finish_circuit()
print('rotate3x12 + cswap3x12 toffoli:', c.operation_statistics().num_toffoli_like)
sim = km.Simulator(batch_size=2, seed=1)
sim.use_same_registers_as(c)
# set tau = [5,7,9], sigma=[1,2,3], flag=1
for i, v in enumerate([5,7,9]):
    for bit in range(12):
        if (v>>bit)&1:
            sim.write_within_shot(tau[i][bit], 0, True); sim.write_within_shot(tau[i][bit], 1, True)
for i, v in enumerate([1,2,3]):
    for bit in range(12):
        if (v>>bit)&1:
            sim.write_within_shot(sigma[i][bit], 0, True); sim.write_within_shot(sigma[i][bit], 1, True)
sim.write_within_shot(flag, 0, True)
sim.write_within_shot(flag, 1, True)
sim.do(c)
def read(reg, shot):
    return sum((1<<bit) for bit in range(12) if sim.read_within_shot(reg[bit], shot))
for s in range(2):
    print(f'shot{s}: sigma={[read(sigma[i],s) for i in range(3)]} tau={[read(tau[i],s) for i in range(3)]} flag={sim.read_within_shot(flag,s)}')

# gf2_imul is constant mult: check cost
b = km.CircuitBuilder()
t = b.alloc_qubits(12)
b.gf2_imul(0xABC, target=t, field=gf)
c = b.finish_circuit()
print('gf2_imul const toffoli:', c.operation_statistics().num_toffoli_like, 'ops:', c.operation_statistics().num_instructions)

# iadd by 1 on 8 bits cost again, single
b = km.CircuitBuilder()
t = b.alloc_qubits(8)
b.iadd(1, target=t)
c = b.finish_circuit()
print('iadd(1, 8-bit) toffoli:', c.operation_statistics().num_toffoli_like)

# check register_data / num_registers for anonymous bits
b = km.CircuitBuilder()
s = b.create_quantum_register(12, name='S0')
anon = b.alloc_qubits(12)
anonb = b.alloc_bits(5)
c = b.finish_circuit()
print('register_data keys:', list(c.register_data.keys()))
print('num_registers:', c.num_registers, 'num_qubits:', c.num_qubits, 'num_bits:', c.num_bits)
EOF
