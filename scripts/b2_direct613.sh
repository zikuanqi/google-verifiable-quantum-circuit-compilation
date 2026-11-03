#!/bin/bash
# b2: direct bezout with the printed values (case 613).
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
    if trial < 613: continue
    A = [1989, 2714, 483, 991, 1511, 2759, 0,0,0,0,0,0,0,0,0,0,0,0]
    B = [3226, 3219, 734, 453, 3736, 2162, 1898, 1690, 41, 2846, 3896, 0,0,0,0,0,0,0]
    prod = [x ^ y for x, y in zip(pmul(A, E, 40), pmul(B, O, 40))]
    nz = [i for i, x in enumerate(prod) if x]
    print('E:', E[:dE+1])
    print('O:', O[:dO+1])
    print('bezout nz:', nz[:8], 'count:', len(nz), 'const:', prod[0])
EOF
