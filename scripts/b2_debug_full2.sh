#!/bin/bash
# b2: instrument the real-sample pipeline step by step.
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
cutoffs = [min(r, 35) for r in range(1, 71)]
s = dist.sample(rng)
S0 = s['syndromes'][:]
errs = s['errors']
print('weight:', s['weight'])

# forward BM
bm = M.BMState(gf, cutoffs, n=70)
for r in range(1, 71):
    bm.round_fwd(r, S0)
sig = bm.sigma
print('deg sigma (pre-normalize):', M._poly_deg(sig[:36]), 'sigma0:', sig[0])

# roots check
ginv = gf.inv(2); beta = 1; roots = []
for j in range(4095):
    if j: beta = gf.mul(beta, ginv)
    if M.eval_poly(sig[:36], beta, gf) == 0:
        roots.append(j)
print('roots match:', sorted(roots) == sorted(errs.keys()), 'nroots:', len(roots))

# normalize
delta = gf.inv(sig[0])
sign = [gf.mul(x, delta) for x in sig[:36]]
print('deg sigma (normalized):', M._poly_deg(sign[:36]), 'sigma0n:', sign[0])

# product
S = S0[:]
M.omega_product(S, sign, 34, gf, reverse=False)
tail_nz = [i for i in range(36, 70) if S[i]]
print('tail nonzero indices:', tail_nz[:8], 'count:', len(tail_nz))

# EEA
E, O = M.sigma_to_eo(sign, gf)
dE, dO = M._poly_deg(E), M._poly_deg(O)
print('dE, dO:', dE, dO)
r0 = M._align(E); r1 = M._align(O)
d0, d1 = dE, dO
if d0 < d1:
    r0, r1 = r1, r0; d0, d1 = d1, d0
r0f, r1f, s0f, s1f, t0f, t1f, d0f, d1f, a_f = M.eea_eliminate(r0, r1, d0, d1, gf)
a = a_f % 18
Aw = [s0f[(a+i) % 18] for i in range(18)]
Bw = [t0f[(a+i) % 18] for i in range(17)]
cg = r0f[a]
prod = [0]*35
for i in range(18):
    for j in range(18):
        prod[i+j] ^= gf.mul(Aw[i], E[j])
for i in range(17):
    for j in range(18):
        prod[i+j] ^= gf.mul(Bw[i], O[j])
nz = [i for i, x in enumerate(prod) if x]
print('EEA: d1f:', d1f, 'cg:', cg, 'bezout nz:', nz[:5], 'degA:', M._poly_deg(Aw[:18]), 'degB:', M._poly_deg(Bw[:17]))
EOF
