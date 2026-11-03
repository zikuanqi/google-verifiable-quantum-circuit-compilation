#!/bin/bash
# b2: check the reconstruction's invariant at each fire (case 613).
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
    if trial < 613:
        continue
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
    W = 18
    s0 = [0]*W; t0 = [0]*W; s1 = [0]*W; t1 = [0]*W
    s0[(17-dE) % W] = 1
    t1[(17-dO) % W] = 1
    c = 0; a = 17 - dE; done = dO < 0
    dd0, dd1 = dE, dO
    r0r = M._align(E); r1r = M._align(O)
    for it in range(35):
        q = qs[it]
        for i in range(W):
            s0[i] ^= gf.mul(q, s1[i])
            t0[i] ^= gf.mul(q, t1[i])
        s0 = [s0[-1]] + s0[:-1]
        t0 = [t0[-1]] + t0[:-1]
        # r0's replay for the check
        for i in range(18):
            r0r[i] ^= gf.mul(q, r1r[i])
        r0r = [r0r[-1]] + r0r[:-1]
        c += 1; a += 1
        if fires[it]:
            # invariant check: s0*E + t0*O == r0r (the poly part)
            rp = [r0r[(a + i) % 18] for i in range(18)]
            pr = [x ^ y for x, y in zip(pmul(s0, E, 40), pmul(t0, O, 40))]
            ok = all(pr[i] == (rp[i] if i < 18 else 0) for i in range(40))
            print(f'  fire at it {it}: c={c} a={a} invariant_ok={ok}')
            if not ok:
                print('    pr[:8]:', pr[:8])
                print('    rp[:8]:', rp[:8])
                break
            s0, s1 = s1, s0
            t0, t1 = t1, t0
            r0r, r1r = r1r, r0r
            dd0, dd1 = dd1, dd0 - c
            c = 0; a = 17 - dd0
            done = dd1 < 0
        elif fires[it]:
            pass
EOF
