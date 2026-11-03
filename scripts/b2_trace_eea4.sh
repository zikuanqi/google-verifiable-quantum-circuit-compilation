#!/bin/bash
# b2: trace the EEA on the real 14/13 case.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
import b2_model as M

gf = M.GF12()
rng = np.random.default_rng(7)
# build a random coprime pair of degrees 14, 13
dE, dO = 14, 13
E = [0]*18; O = [0]*18
E[0] = int(rng.integers(1, 4096))
for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
O[0] = int(rng.integers(1, 4096))
r0 = M._align(E); r1 = M._align(O)
d0, d1 = dE, dO

# manual trace with s/t
s0 = [0]*18; t0 = [0]*18; s1 = [0]*18; t1 = [0]*18
s0[17-d0] = 1; t1[17-d1] = 1
c = 0; a = 17-d0; done = d1 < 0
for it in range(35):
    q = gf.mul(r0[17], gf.inv(r1[17]))
    for i in range(18):
        r0[i] ^= gf.mul(q, r1[i])
        s0[i] ^= gf.mul(q, s1[i])
        t0[i] ^= gf.mul(q, t1[i])
    r0[17] ^= q
    r0 = [r0[-1]] + r0[:-1]
    s0 = [s0[-1]] + s0[:-1]
    t0 = [t0[-1]] + t0[:-1]
    c += 1; a += 1
    if not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0):
        r0, r1 = r1, r0
        s0, s1 = s1, s0
        t0, t1 = t1, t0
        d0, d1 = d1, d0 - c
        c = 0; a = 17 - d0
        done = d1 < 0
        print(f'  swap at it {it}: new d0,d1 = {d0},{d1} a={a} degs0={M._poly_deg(s0)} degt0={M._poly_deg(t0)}')
print('final: d0,d1 =', d0, d1, 'a =', a, 'degs0 =', M._poly_deg(s0), 'degt0 =', M._poly_deg(t0))
a %= 18
Aw = [s0[(a+i) % 18] for i in range(18)]
Bw = [t0[(a+i) % 18] for i in range(17)]
cg = r0[a]
prod = [0]*35
for i in range(18):
    for j in range(18):
        prod[i+j] ^= gf.mul(Aw[i], E[j])
for i in range(17):
    for j in range(18):
        prod[i+j] ^= gf.mul(Bw[i], O[j])
nz = [i for i, x in enumerate(prod) if x]
print('cg:', cg, 'bezout nz:', nz[:6], 'degA:', M._poly_deg(Aw), 'degB:', M._poly_deg(Bw))
print('r0 array:', r0)
EOF
