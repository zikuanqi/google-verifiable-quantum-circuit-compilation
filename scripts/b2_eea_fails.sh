#!/bin/bash
# b2: inspect the 6/5000 reconstruction failures.
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
def test(W, dE, dO, E, O):
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
    s0 = [0]*W; t0 = [0]*W; s1 = [0]*W; t1 = [0]*W
    s0[(17-dE) % W] = 1
    t1[(17-dO) % W] = 1
    c = 0; a = 17 - dE; done = dO < 0
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
            done = dd1 < 0
    a %= W
    Aw = [s0[(a+i) % W] for i in range(W)]
    Bw = [t0[(a+i) % W] for i in range(W)]
    prod = [x ^ y for x, y in zip(pmul(Aw, E, 40), pmul(Bw, O, 40))]
    nz = [i for i, x in enumerate(prod) if x]
    return (len(nz) == 1 and nz[0] == 0), nz, d1, a

rng = np.random.default_rng(9)
shown = 0
for trial in range(5000):
    dE = int(rng.integers(2, 18)); dO = int(rng.integers(1, dE+1))
    E = [0]*18; O = [0]*18
    E[0] = int(rng.integers(1, 4096))
    for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
    for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
    O[0] = int(rng.integers(1, 4096))
    ok, nz, d1f, af = test(18, dE, dO, E, O)
    if not ok and shown < 3:
        # gcd degree check
        P, Q = E[:], O[:]
        while M._poly_deg(Q) >= 0:
            dP, dQ = M._poly_deg(P), M._poly_deg(Q)
            if dP < dQ:
                P, Q = Q, P; dP, dQ = dQ, dP
            lc = Q[dQ]
            for i in range(dP, dQ-1, -1):
                if P[i]:
                    cf = gf.mul(P[i], gf.inv(lc))
                    for j in range(dQ+1):
                        P[i-dQ+j] ^= gf.mul(cf, Q[j])
            while P and P[-1] == 0: P.pop()
            P, Q = Q, P
        print(f'FAIL trial {trial}: dE={dE} dO={dO} gcd_deg={M._poly_deg(P)} d1f={d1f} a={af} nz={nz[:6]}')
        shown += 1
EOF
