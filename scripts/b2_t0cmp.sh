#!/bin/bash
# b2: compare t0/B between array and poly models (case 613).
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

rng = np.random.default_rng(9)
for trial in range(614):
    dE = int(rng.integers(2, 18)); dO = int(rng.integers(1, dE+1))
    E = [0]*18; O = [0]*18
    E[0] = int(rng.integers(1, 4096))
    for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
    for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
    O[0] = int(rng.integers(1, 4096))
    if trial < 613: continue
    r0 = M._align(E); r1 = M._align(O)
    d0, d1 = dE, dO
    qs, fires = [], []
    c = 0; a = 17 - d0; done = d1 < 0
    for it in range(35):
        q = gf.mul(r0[17], gf.inv(r1[17]))
        qs.append(q)
        for i in range(18):
            r0[i] ^= gf.mul(q, r1[i])
        r0 = [r0[-1]] + r0[:-1]
        c += 1; a += 1
        fire = (not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0))
        if fire:
            r0, r1 = r1, r0
            d0, d1 = d1, d0 - c
            c = 0; a = 17 - d0
            done = d1 < 0
        fires.append(fire)
    # poly
    s0p = [1]; s1p = [0]; t0p = [0]; t1p = [1]
    c = 0; dd0, dd1 = dE, dO
    for it in range(35):
        q = qs[it]
        sh = dd0 - dd1 - c
        ns0 = s0p + [0]*max(0, len(s1p)+sh-len(s0p))
        for i, v in enumerate(s1p):
            if v and 0 <= i+sh < len(ns0): ns0[i+sh] ^= gf.mul(q, v)
        nt0 = t0p + [0]*max(0, len(t1p)+sh-len(t0p))
        for i, v in enumerate(t1p):
            if v and 0 <= i+sh < len(nt0): nt0[i+sh] ^= gf.mul(q, v)
        s0p, t0p = ns0, nt0
        c += 1
        if fires[it]:
            s0p, s1p = s1p, s0p
            t0p, t1p = t1p, t0p
            dd0, dd1 = dd1, dd0 - c
            c = 0
    # array W=18
    W = 18
    s0 = [0]*W; t0 = [0]*W; s1 = [0]*W; t1 = [0]*W
    s0[(17-dE) % W] = 1
    t1[(17-dO) % W] = 1
    c = 0; a = 17 - dE
    dd0, dd1 = dE, dO
    for it in range(35):
        q = qs[it]
        for i in range(W):
            s0[i] ^= gf.mul(q, s1[i])
            t0[i] ^= gf.mul(q, t1[i])
        s0 = [s0[-1]] + s0[:-1]
        t0 = [t0[-1]] + t0[:-1]
        c += 1; a += 1
        if fires[it]:
            s0, s1 = s1, s0
            t0, t1 = t1, t0
            dd0, dd1 = dd1, dd0 - c
            c = 0; a = 17 - dd0
    a %= W
    Aw = [s0[(a+i) % W] for i in range(W)]
    Bw = [t0[(a+i) % W] for i in range(W)]
    print('poly: degs0:', M._poly_deg(s0p), 'degt0:', M._poly_deg(t0p))
    print('array: degA:', M._poly_deg(Aw), 'degB:', M._poly_deg(Bw), 'a:', a)
    print('poly t0[:12]:', t0p[:12])
    print('array Bw[:12]:', Bw[:12])
    print('poly s0[:8]:', s0p[:8])
    print('array Aw[:8]:', Aw[:8])
EOF
