#!/bin/bash
# b2: direct diff of the two traces' uneliminate calls.
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
r0f, r1f, qs, fires, d0f, d1f, a_f = M.eea_eliminate(r0, r1, d0, d1, gf)
print('fires:', fires)
print('nz q count:', sum(1 for q in qs if q))
r0u, r1u = M.eea_uneliminate(r0f, r1f, qs, fires, gf)
print('restore:', r0u == r0, r1u == r1)
for i in range(18):
    if r0u[i] != r0[i]:
        print('  first r0 diff at', i, r0u[i], 'vs', r0[i])
        break
for i in range(18):
    if r1u[i] != r1[i]:
        print('  first r1 diff at', i, r1u[i], 'vs', r1[i])
        break
EOF
