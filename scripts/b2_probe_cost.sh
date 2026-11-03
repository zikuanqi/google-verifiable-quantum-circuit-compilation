#!/bin/bash
# b2: cost probe for kickmix primitives (12-bit field).
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
$PY - <<'EOF'
import kickmix as km

gf = km.GF2Field(12)
print('modulus:', hex(gf.modulus), 'primitive:', gf.primitive_element, 'irred:', gf.is_irreducible(gf.modulus))
# quick sanity: mul
a, b = 0x5A5, 0x3C3
print('mul check:', gf.mul(a,b), 'invert check:', gf.mul(5, gf.invert(5)))

def stats_of(build):
    bld = km.CircuitBuilder()
    build(bld)
    c = bld.finish_circuit()
    st = c.operation_statistics()
    return st

def show(name, build):
    st = stats_of(build)
    print(f'{name}: toffoli_like={st.num_toffoli_like} n_ops={st.num_instructions} nqubits={st.num_qubits if hasattr(st,"num_qubits") else "?"}')
    return st

# 1. single gates
show('X', lambda b: b.x(b.alloc_qubits(1)))
show('CX', lambda b: (lambda q: b.cx(q[0], q[1]))(b.alloc_qubits(2)))
show('CCX', lambda b: (lambda q: b.ccx(q[0], q[1], q[2]))(b.alloc_qubits(3)))
show('CSWAP', lambda b: (lambda q: b.cswap(q[0], q[1], q[2]))(b.alloc_qubits(3)))
show('SWAP', lambda b: (lambda q: b.swap(q[0], q[1]))(b.alloc_qubits(2)))
show('Z', lambda b: b.z(b.alloc_qubits(1)[0]))
show('multiX(3c)', lambda b: (lambda q: b.multi_controlled_x(q, b.alloc_qubits(1)[0]))(b.alloc_qubits(3)))

# 2. GF mul
def build_mul(b):
    x = b.alloc_qubits(12); y = b.alloc_qubits(12)
    t = b.init_gf2_mul(x, y, field=gf)
    b.del_gf2_mul(x, y, target=t, field=gf)
show('gf2_mul(12) init+del', build_mul)

def build_ixormul(b):
    x = b.alloc_qubits(12); y = b.alloc_qubits(12); t = b.alloc_qubits(12)
    b.ixor_gf2_mul(x, y, target=t, field=gf)
    b.ixor_gf2_mul(x, y, target=t, field=gf)
show('ixor_gf2_mul(12) x2', build_ixormul)

# 3. GF inverse
def build_inv(b):
    x = b.alloc_qubits(12)
    t, scaff = b.init_gf2_inverse_with_scaffold(x, field=gf)
    print('  inv scaffold len:', len(scaff))
    b.del_gf2_inverse_with_scaffold(x, target=t, scaffold=scaff, field=gf)
show('gf2_inv(12) scaffolded', build_inv)

def build_inv2(b):
    x = b.alloc_qubits(12)
    t = b.init_gf2_inverse(x, field=gf)
    b.del_gf2_inverse(x, target=t, field=gf)
show('gf2_inv(12) auto-clean', build_inv2)

# 4. GF div
def build_div(b):
    x = b.alloc_qubits(12); y = b.alloc_qubits(12)
    t, scaff = b.init_gf2_div_with_scaffold(x, y, field=gf)
    print('  div scaffold len:', len(scaff))
    b.del_gf2_div_with_scaffold(x, y, target=t, scaffold=scaff, field=gf)
show('gf2_div(12) scaffolded', build_div)

# 5. flip_if_equal (12-bit vs 0)
def build_feq(b):
    x = b.alloc_qubits(12); t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x, 0, target=t)
    b.flip_if_equal(x, 0, target=t)
show('flip_if_equal(12,0) x2', build_feq)

# 6. rotate by 1 on 36 qubits
def build_rot(b):
    q = b.alloc_qubits(36)
    b.left_rotate(q, 1)
    b.left_rotate(q, 1)
show('left_rotate(36,1) x2', build_rot)

# 7. iadd 1 on 8-bit counter
def build_iadd(b):
    t = b.alloc_qubits(8)
    b.iadd(1, target=t)
    b.isub(1, target=t)
show('iadd1+isub1 on 8 bits', build_iadd)

# 8. cswap broadcast 12 pairs
def build_cswap12(b):
    a = b.alloc_qubits(12); c_ = b.alloc_qubits(12); ctrl = b.alloc_qubits(1)[0]
    b.cswap(ctrl, a, c_)
show('cswap 12 pairs', build_cswap12)

# 9. HMR semantics + phase
st = stats_of(lambda b: None)
print('empty circuit toffoli:', st.num_toffoli_like)

# Phase semantics test
bld = km.CircuitBuilder()
q = bld.alloc_qubits(2)
out = bld.alloc_bits(1)
bld.x(q[0])            # q0 = 1
bld.hmr(q[0], out[0])  # X-basis measure
c = bld.finish_circuit()
sim = km.Simulator(batch_size=8, seed=12345)
sim.use_same_registers_as(c)
sim.do(c)
print('HMR: phase_flipped per shot:', [sim.read_shot_phase(i) for i in range(8)])
print('HMR: q0 after:', [sim.read_within_shot(km.q(0), i) for i in range(8)])
print('HMR: out bit:', [sim.read_within_shot(km.b(0), i) for i in range(8)])

# num_qubits with free: peak or total?
bld2 = km.CircuitBuilder()
q1 = bld2.alloc_qubits(100)
bld2.free(q1)
q2 = bld2.alloc_qubits(50)
c2 = bld2.finish_circuit()
print('num_qubits after alloc100/free/alloc50:', c2.num_qubits, '(peak=100, live=50)')

# what keys does OperationStatistics see?
st = stats_of(lambda b: (lambda q: (b.ccx(q[0],q[1],q[2]), b.cswap(q[3],q[4],q[5])))(b.alloc_qubits(6)))
print('op stat keys:', list(st.keys()), 'toffoli_like:', st.num_toffoli_like)
EOF
