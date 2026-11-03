#!/bin/bash
# b2: debug EEA - check d1f and trace s/t for a failing case.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
from b2_model import GF12, _align, _poly_deg, eea_eliminate

gf = GF12()
rng = np.random.default_rng(42)
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
    # invariant check: r0f (as array) == s0f*E_eff_aligned + t0f*O_eff_aligned?
    # compute: s0f represents s0_poly with alignment 17-d0f... simpler: check
    # the invariant r0 = s0*E + t0*O in array form at the end:
    prod = [0]*18
    Ea = _align(E_eff); Oa = _align(O_eff)
    for i in range(18):
        for j in range(18):
            if s0f[i] and Ea[j]: prod[(i+j) % 18] ^= 0  # placeholder
    print(f'trial {trial}: dE={dE} dO={dO} d1f={d1f} d0f={d0f} dA={_poly_deg(s0f)} dB={_poly_deg(t0f)}')
    if trial == 0:
        print('   s0f =', s0f)
        print('   t0f =', t0f)
        print('   r0f =', r0f)
        print('   s1f =', s1f)
        print('   t1f =', t1f)
EOF
