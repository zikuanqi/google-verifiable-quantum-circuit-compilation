#!/bin/bash
# b2: check the Bezout invariant mid-run for the 14/13 EEA.
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
dE, dO = 14, 13
E = [0]*18; O = [0]*18
E[0] = int(rng.integers(1, 4096))
for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
O[0] = int(rng.integers(1, 4096))
Ea = M._align(E); Oa = M._align(O)

def arr_to_poly(arr, a, W):
    return [arr[(a + i) % W] for i in range(W)]

def pmul(P, Q, n):
    R = [0]*n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i+j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

r0 = [x for x in Ea]; r1 = [x for x in Oa]
d0, d1 = dE, dO
W = 19
s0 = [0]*W; t0 = [0]*W; s1 = [0]*W; t1 = [0]*W
s0[(17-d0) % W] = 1; t1[(17-d1) % W] = 1
c = 0; a = 17-d0; done = d1 < 0
for it in range(35):
    q = gf.mul(r0[17], gf.inv(r1[17]))
    for i in range(18):
        r0[i] ^= gf.mul(q, r1[i])
    for i in range(W):
        s0[i] ^= gf.mul(q, s1[i])
        t0[i] ^= gf.mul(q, t1[i])
    r0[17] ^= q
    r0 = [r0[-1]] + r0[:-1]
    s0 = [s0[-1]] + s0[:-1]
    t0 = [t0[-1]] + t0[:-1]
    c += 1; a += 1
    if not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0):
        # check invariant before swap: r0 == s0*E + t0*O (poly level)
        s0p = arr_to_poly(s0, a, W); t0p = arr_to_poly(t0, a, W)
        r0p = arr_to_poly(r0, a, 18)
        lhs = [x ^ y for x, y in zip(pmul(s0p, E, 40), pmul(t0p, O, 40))]
        rhs = r0p
        ok = all(lhs[i] == (rhs[i] if i < 18 else 0) for i in range(40))
        print(f'  swap at it {it} (d0={d0},d1={d1}): invariant_ok={ok} degs0={M._poly_deg(s0p)} degt0={M._poly_deg(t0p)} degr0={M._poly_deg(r0p)}')
        r0, r1 = r1, r0
        s0, s1 = s1, s0
        t0, t1 = t1, t0
        d0, d1 = d1, d0 - c
        c = 0; a = 17 - d0
        done = d1 < 0
print('final d0,d1 =', d0, d1, 'a =', a)
s0p = arr_to_poly(s0, a, W); t0p = arr_to_poly(t0, a, W); r0p = arr_to_poly(r0, a, 18)
lhs = [x ^ y for x, y in zip(pmul(s0p, E, 40), pmul(t0p, O, 40))]
print('final invariant:', all(lhs[i] == (r0p[i] if i < 18 else 0) for i in range(40)))
print('final degs0:', M._poly_deg(s0p), 'degt0:', M._poly_deg(t0p), 'degr0:', M._poly_deg(r0p))
EOF
