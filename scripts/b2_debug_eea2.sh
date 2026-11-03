#!/bin/bash
# b2: debug the new EEA with live Bezout.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
from b2_model import (GF12, _align, _poly_deg, _poly_zero,
                      eea_eliminate, eea_uneliminate)

gf = GF12()

def pmul(P, Q, n):
    R = [0] * n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i + j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

rng = np.random.default_rng(42)
ok_all = True
for trial in range(8):
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
    r0f, r1f, s0f, s1f, t0f, t1f, d0f, d1f = eea_eliminate(r0, r1, d0, d1, gf)
    # extract A, B right-aligned
    dA = _poly_deg(s0f); dB = _poly_deg(t0f)
    Aw = [s0f[17-dA+i] if i <= dA else 0 for i in range(18)]
    Bw = [t0f[17-dB+i] if i <= dB else 0 for i in range(17)]
    prod = [0]*35
    for i in range(18):
        for j in range(18):
            prod[i+j] ^= gf.mul(Aw[i], E_eff[j])
    for i in range(17):
        for j in range(18):
            prod[i+j] ^= gf.mul(Bw[i], O_eff[j])
    nz = [i for i, x in enumerate(prod) if x]
    bez = (len(nz) == 1 and nz[0] == 0)
    r0u, r1u, s0u, s1u, t0u, t1u = eea_uneliminate(r0f, r1f, s0f, s1f, t0f, t1f, d0f, d1f, gf)
    rev = (r0u == r0 and r1u == r1 and s0u == [0]*17+[1] and s1u == [0]*18 and t0u == [0]*18)
    print(f'trial {trial}: dE={dE} dO={dO} r1_zero={_poly_zero(r1f)} bezout={bez} nz={nz[:2]} reverse_ok={rev} dA={dA} dB={dB}')
    ok_all &= bez and rev and _poly_zero(r1f)
print('ALL OK:', ok_all)
EOF
