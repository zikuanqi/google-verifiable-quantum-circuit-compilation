#!/bin/bash
# b2: verify EEA + full pipeline on one real sample.
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
mu_star = gf.dual_basis()
s = dist.sample(rng)
r = M.run_one(s, gf, cutoffs, 34, 70, mu_star, 4095)
for k, v in r.items():
    print(f'  {k}: {v}')

# small-case EEA sanity
from b2_model import _align, _poly_deg, eea_eliminate, eea_uneliminate
for trial in range(4):
    dE = int(rng.integers(3, 9)); dO = int(rng.integers(1, dE+1))
    E = [0]*18; O = [0]*18
    E[0] = int(rng.integers(1, 4096))
    for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
    for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
    O[0] = int(rng.integers(1, 4096))
    r0 = _align(E); r1 = _align(O)
    d0, d1 = _poly_deg(E), _poly_deg(O)
    if d0 < d1:
        r0, r1 = r1, r0; d0, d1 = d1, d0
    E_eff = E if d0 >= d1 else O
    O_eff = O if d0 >= d1 else E
    r0f, r1f, s0f, s1f, t0f, t1f, d0f, d1f, a_f = eea_eliminate(r0, r1, d0, d1, gf)
    a = a_f % 18
    Aw = [s0f[(a+i) % 18] for i in range(18)]
    Bw = [t0f[(a+i) % 18] for i in range(17)]
    cg = r0f[a]
    prod = [0]*35
    for i in range(18):
        for j in range(18):
            prod[i+j] ^= gf.mul(Aw[i], E_eff[j])
    for i in range(17):
        for j in range(18):
            prod[i+j] ^= gf.mul(Bw[i], O_eff[j])
    nz = [i for i, x in enumerate(prod) if x]
    bez = (len(nz) == 1 and nz[0] == 0 and prod[0] == cg)
    print(f'  small {trial}: d1f={d1f} cg={cg} bezout={bez} nz={nz[:3]} degA={_poly_deg(Aw)} degB={_poly_deg(Bw)}')
EOF
