#!/bin/bash
# b2: manual step trace of the uneliminate.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
import b2_model as M
gf = M.GF12()
dist = M.DQIDistribution(gf=gf)
rng = np.random.default_rng(3)
s = dist.sample(rng)
S0 = s['syndromes'][:]
cutoffs = [min(r, 35) for r in range(1, 71)]
bm = M.BMState(gf, cutoffs, n=70)
for r in range(1, 71):
    bm.round_fwd(r, S0)
sig = bm.sigma
delta = gf.inv(sig[0])
sign = [gf.mul(x, delta) for x in sig[:36]]
E, O = M.sigma_to_eo(sign, gf)
r0 = M._align(E); r1 = M._align(O)
d0, d1 = M._poly_deg(E), M._poly_deg(O)
# manual forward with snapshots
snap = []
c = 0
for it in range(35):
    q = gf.mul(r0[17], gf.inv(r1[17]))
    for i in range(18):
        r0[i] ^= gf.mul(q, r1[i])
    r0 = [r0[-1]] + r0[:-1]
    c += 1
    fire = (not (d1 < 0) and c >= d0 - d1 + 1 and (r0[17] != 0 or d1 == 0))
    if fire:
        r0, r1 = r1, r0
        d0, d1 = d1, d0 - c
        c = 0
    snap.append(([x for x in r0], [x for x in r1], q, fire))
# reverse, comparing with snapshots
r0u, r1u = [x for x in snap[-1][0]], [x for x in snap[-1][1]]
for it in range(34, -1, -1):
    if snap[it][3]:
        r0u, r1u = r1u, r0u
    q = snap[it][2]
    r0u = r0u[1:] + [r0u[0]]
    for i in range(18):
        r0u[i] ^= gf.mul(q, r1u[i])
    if it > 0:
        expect = snap[it-1][0]
        if r0u != expect:
            print(f'it {it}: MISMATCH after step vs snapshot it-1')
            for i in range(18):
                if r0u[i] != expect[i]:
                    print('   word', i, r0u[i], 'vs', expect[i])
            break
else:
    print('all steps matched')
EOF
