#!/bin/bash
# b2: debug the EEA eliminate+reconstruct.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
from b2_model import (GF12, _align, _poly_deg, _poly_zero,
                      eea_eliminate, eea_reconstruct, eea_uneliminate)

gf = GF12()

def pmul(P, Q, n):
    """polynomial product truncated to n words."""
    R = [0] * n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i + j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

def padd(P, Q):
    return [a ^ b for a, b in zip(P, Q)]

rng = np.random.default_rng(42)
for trial in range(6):
    # random coprime pair E, O of degree <= 8
    dE = int(rng.integers(3, 9)); dO = int(rng.integers(1, dE+1))
    E = [0]*18; O = [0]*18
    E[0] = int(rng.integers(1, 4096)); O[0] = 0
    for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
    for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
    O[0] = int(rng.integers(1, 4096))
    if dO == 0:
        continue
    r0 = _align(E); r1 = _align(O)
    d0, d1 = _poly_deg(E), _poly_deg(O)
    if d0 < d1:
        r0, r1 = r1, r0; d0, d1 = d1, d0
    r0s, r1s = [x for x in r0], [x for x in r1]
    r0f, r1f, d0f, d1f, qseq = eea_eliminate(r0, r1, d0, d1, gf)
    # check remainder zero and gcd constant
    print(f'trial {trial}: dE={dE} dO={dO} r1_zero={_poly_zero(r1f)} nq_nonzero={sum(1 for q in qseq if q)}')
    # gcd = r0f (aligned)
    dg = _poly_deg(r0f)
    gcd_deg = dg  # should be 0
    Aw, Bw, dA, dB = eea_reconstruct(qseq, gf, 18, 17)
    # Bezout check: A*E + B*O == gcd? (up to scaling)
    prod = padd(pmul(Aw[:18], E, 35), pmul(Bw[:17], O, 35))
    nz = [i for i, x in enumerate(prod) if x]
    # the gcd is a constant; check prod == [c, 0, 0, ...]
    ok = (len(nz) == 0) or (nz == [0])
    print(f'   bezout_ok={ok} prod_nonzero_deg={nz[:3]} dA={dA} dB={dB}')
    if not ok:
        print('   A=', Aw[:dA+1])
        print('   B=', Bw[:dB+1])
        # try the alternative recurrence: newA[i] = q*A[i-1] + B[i-1] with z^-1 style
        A2 = [0]*18; B2 = [0]*18; A2[0] = 1
        for q in qseq:
            if q == 0: continue
            newA = [0]*18
            for i in range(0, 17):
                newA[i] = gf.mul(q, A2[i+1]) ^ B2[i+1]   # z^-1 style: A_new[i] = q*A[i+1]+B[i+1]
            B2 = A2[:]; A2 = newA
        prod2 = padd(pmul(A2[:18], E, 35), pmul(B2[:18], O, 35))
        nz2 = [i for i, x in enumerate(prod2) if x]
        print('   alt (z^-1) bezout_ok=', (len(nz2)==0) or (nz2==[0]), ' nz=', nz2[:3])
        break
EOF
