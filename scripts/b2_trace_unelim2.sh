#!/bin/bash
# b2: trace the uneliminate on the real 14/13 sample.
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
if d0 < d1:
    r0, r1 = r1, r0; d0, d1 = d1, d0
r0i, r1i = [x for x in r0], [x for x in r1]
r0f, r1f, qs, fires, d0f, d1f, a_f = M.eea_eliminate(r0, r1, d0, d1, gf)
r0u, r1u = M.eea_uneliminate(r0f, r1f, qs, fires, gf)
print('d0,d1 =', d0, d1, 'n_fires =', sum(fires), 'restore:', r0u == r0i, r1u == r1i)
# first divergence
for i in range(18):
    if r0u[i] != r0i[i]:
        print('first r0 diff at word', i, ':', r0u[i], 'vs', r0i[i])
        break
EOF
