#!/bin/bash
# b2: measure max alignment of s/t arrays over many random EEA runs.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
import b2_model as M

gf = M.GF12()
rng = np.random.default_rng(123)
max_a_s0 = 0; max_a_s1 = 0; max_a_t0 = 0
for trial in range(3000):
    dE = int(rng.integers(2, 18)); dO = int(rng.integers(1, dE+1))
    E = [0]*18; O = [0]*18
    E[0] = int(rng.integers(1, 4096))
    for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
    for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
    O[0] = int(rng.integers(1, 4096))
    r0 = M._align(E); r1 = M._align(O)
    d0, d1 = dE, dO
    if d0 < d1:
        r0, r1 = r1, r0; d0, d1 = d1, d0
    W = 19
    s0 = [0]*W; t0 = [0]*W; s1 = [0]*W; t1 = [0]*W
    s0[17-d0] = 1; t1[17-d1] = 1
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
        max_a_s0 = max(max_a_s0, a + M._poly_deg(s0))
        max_a_s1 = max(max_a_s1, (17-d1) + M._poly_deg(s1))
        max_a_t0 = max(max_a_t0, a + M._poly_deg(t0))
        if not done and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0):
            r0, r1 = r1, r0
            s0, s1 = s1, s0
            t0, t1 = t1, t0
            d0, d1 = d1, d0 - c
            c = 0; a = 17 - d0
            done = d1 < 0
print('max a+deg over 3000 runs: s0:', max_a_s0, 's1:', max_a_s1, 't0:', max_a_t0)
EOF
