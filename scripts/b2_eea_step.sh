#!/bin/bash
# b2: find the first invariant failure in the EEA (step by step).
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

def pmul(P, Q, n):
    R = [0]*n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i+j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

r0 = M._align(E); r1 = M._align(O)
d0, d1 = dE, dO
s0 = [1]; s1 = [0]; t0 = [0]; t1 = [1]
c = 0; done = d1 < 0
for it in range(35):
    # invariant BEFORE the update
    a = 17 - d0 + c
    r0p = [r0[(a + i) % 18] for i in range(18)]
    lhs = [x ^ y for x, y in zip(pmul(s0, E, 40), pmul(t0, O, 40))]
    ok = all(lhs[i] == (r0p[i] if i < 18 else 0) for i in range(40))
    if not ok:
        print(f'it {it} PRE-update invariant FAILED: d0={d0} d1={d1} c={c} a={a}')
        print('  lhs[:6]:', lhs[:6], ' r0p[:6]:', r0p[:6])
        print('  lhs[12:18]:', lhs[12:18], ' r0p[12:18]:', r0p[12:18])
        break
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
else:
    print('all pre-update invariants hold')
EOF
