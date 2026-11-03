#!/bin/bash
# b2: test the POLY-level EEA model's bezout directly.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
import b2_model as M

gf = M.GF12()

def pmul(P, Q, n):
    R = [0]*n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i+j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

rng = np.random.default_rng(7)
dE, dO = 14, 13
E = [0]*18; O = [0]*18
E[0] = int(rng.integers(1, 4096))
for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
O[0] = int(rng.integers(1, 4096))

r0 = M._align(E); r1 = M._align(O)
d0, d1 = dE, dO
s0 = [1]; s1 = [0]; t0 = [0]; t1 = [1]
c = 0; done = d1 < 0
for it in range(35):
    q = gf.mul(r0[17], gf.inv(r1[17]))
    for i in range(18):
        r0[i] ^= gf.mul(q, r1[i])
    r0[17] ^= q
    r0 = [r0[-1]] + r0[:-1]
    sh = d0 - d1 - c
    ns0 = s0 + [0]*max(0, len(s1)+sh-len(s0))
    for i, v in enumerate(s1):
        if v and 0 <= i+sh < len(ns0):
            ns0[i+sh] ^= gf.mul(q, v)
    nt0 = t0 + [0]*max(0, len(t1)+sh-len(t0))
    for i, v in enumerate(t1):
        if v and 0 <= i+sh < len(nt0):
            nt0[i+sh] ^= gf.mul(q, v)
    s0, t0 = ns0, nt0
    c += 1
    if not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0):
        r0, r1 = r1, r0
        s0, s1 = s1, s0
        t0, t1 = t1, t0
        d0, d1 = d1, d0 - c
        c = 0
        done = d1 < 0
# bezout check
prod = [x ^ y for x, y in zip(pmul(s0, E, 40), pmul(t0, O, 40))]
nz = [i for i, x in enumerate(prod) if x]
print('POLY model: degs0:', M._poly_deg(s0), 'degt0:', M._poly_deg(t0),
      'bezout nz:', nz[:6], 'count:', len(nz), 'const:', prod[0] if len(nz) else 0)
# the remainder at the end: r0 = gcd aligned
print('r0:', r0)
EOF
