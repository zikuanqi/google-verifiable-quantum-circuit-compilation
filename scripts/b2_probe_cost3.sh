#!/bin/bash
# b2: probe costs for the builder: gated mults, quantum-shift rotate, small fields.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import kickmix as km

def show(name, build):
    b = km.CircuitBuilder()
    build(b)
    c = b.finish_circuit()
    st = c.operation_statistics()
    print(f'{name}: toff={st.num_toffoli_like} ops={st.num_instructions} nq={c.num_qubits}')

gf12 = km.GF2Field(12)
gf4 = km.GF2Field(4)

# gated ixor_gf2_mul
def g1(b):
    x = b.alloc_qubits(12); y = b.alloc_qubits(12); t = b.alloc_qubits(12); c = b.alloc_qubits(1)[0]
    b.ixor_gf2_mul(x, y, target=t, field=gf12, control=c)
    b.ixor_gf2_mul(x, y, target=t, field=gf12, control=c)
show('ixor_gf2_mul(12) gated x2', g1)

# 4-bit field mults
def g2(b):
    x = b.alloc_qubits(4); y = b.alloc_qubits(4); t = b.alloc_qubits(4)
    b.ixor_gf2_mul(x, y, target=t, field=gf4)
    b.ixor_gf2_mul(x, y, target=t, field=gf4)
show('ixor_gf2_mul(4) x2', g2)

def g3(b):
    x = b.alloc_qubits(4)
    t = b.init_gf2_inverse(x, field=gf4)
    b.del_gf2_inverse(x, target=t, field=gf4)
show('gf2_inv(4) init+del', g3)

# quantum-shift rotate: 19-qubit lane by a 5-bit shift
def g4(b):
    lane = b.alloc_qubits(19)
    sh = b.alloc_qubits(5)
    b.left_rotate(lane, sh)
show('left_rotate(19, 5-bit quantum shift)', g4)

# flip_if_equal on 8-bit and 5-bit
def g5(b):
    x = b.alloc_qubits(8); t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x, 7, target=t)
    b.flip_if_equal(x, 7, target=t)
show('flip_if_equal(8,7) x2', g5)

def g6(b):
    x = b.alloc_qubits(5); t = b.alloc_qubits(1)[0]
    b.flip_if_equal(x, 3, target=t)
    b.flip_if_equal(x, 3, target=t)
show('flip_if_equal(5,3) x2', g6)

# OR of two flags: ctrl = rep | rt via init_and tricks
def g7(b):
    a = b.alloc_qubits(1); bb = b.alloc_qubits(1); out = b.alloc_qubits(1)
    b.cx(a, out[0]); b.cx(bb, out[0])
    b.ccx(a[0], bb[0], out[0])
    b.ccx(a[0], bb[0], out[0])
    b.cx(bb, out[0]); b.cx(a, out[0])
show('OR + unOR of two flags', g7)

# unary_iteration cost over 36 values with 6-bit address
def g8(b):
    addr = b.alloc_qubits(6)
    t = b.alloc_qubits(1)[0]
    with b.unary_iteration(addr) as it:
        for v, mq in it:
            b.cx(mq, t[0])
            b.cx(mq, t[0])
show('unary_iteration(6-bit, full walk, cx x2 per step)', g8)

# controlled rotate (CSWAP-based) of a 36-lane
def g9(b):
    lane = b.alloc_qubits(36)
    c = b.alloc_qubits(1)[0]
    b.left_rotate(lane, 1, control=c)
show('left_rotate(36,1,control) x1', g9)

# ccx with a classical-bit control
def g10(b):
    q1 = b.alloc_qubits(1); q2 = b.alloc_qubits(1); cb = b.alloc_bits(1)
    b.bit_store(cb[0], True)
    b.ccx(q1[0], cb[0], q2[0])
show('ccx(q, b, q)', g10)

# multi_controlled_x with 4 quantum controls
def g11(b):
    cs = b.alloc_qubits(4); t = b.alloc_qubits(1)[0]
    b.multi_controlled_x(cs, t[0])
    b.multi_controlled_x(cs, t[0])
show('multiX(4 controls) x2', g11)
EOF
