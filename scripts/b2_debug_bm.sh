#!/bin/bash
# b2: debug the rescaled BM against textbook BM.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np
import sys
sys.path.insert(0, 'src')
from b2_model import GF12, DQIDistribution, BMState, _poly_deg

gf = GF12()
dist = DQIDistribution(gf=gf)
rng = np.random.default_rng(5)
s = dist.sample(rng)
S = s['syndromes']
errors = s['errors']
print('weight:', s['weight'])

# textbook BM (Massey LFSR synthesis), standard convention
def bm_textbook(S, gf, n=70):
    C = [1] + [0]*n          # connection poly C(z) = 1 + c1 z + ...
    B = [1] + [0]*n
    L = 0
    m = 1
    b = 1
    for r in range(1, n+1):   # process S_r
        d = S[r-1]
        for i in range(1, L+1):
            d ^= gf.mul(C[i], S[r-1-i])
        if d == 0:
            m += 1
        else:
            if 2*L <= r-1:
                T = C[:]
                coef = gf.mul(d, gf.inv(b))
                for i in range(n+1):
                    if i + m <= n:
                        C[i+m] ^= gf.mul(coef, B[i])
                L = r - L
                B = T
                b = d
                m = 1
            else:
                coef = gf.mul(d, gf.inv(b))
                for i in range(n+1):
                    if i + m <= n:
                        C[i+m] ^= gf.mul(coef, B[i])
                m += 1
    return C, L

C, L = bm_textbook(S, gf)
print('textbook L:', L, 'deg:', _poly_deg(C[:36]))
# check: C's roots = error positions
def poly_eval(C, x, gf):
    acc = 0
    for c in reversed(C):
        acc = gf.mul(acc, x) ^ c
    return acc
ginv = gf.inv(2)
beta = 1
roots = []
for j in range(4095):
    if j: beta = gf.mul(beta, ginv)
    if poly_eval(C, beta, gf) == 0:
        roots.append(j)
print('textbook roots:', sorted(roots)[:10], '... count:', len(roots))
print('true error positions:', sorted(errors.keys())[:10], 'count:', len(errors))
print('match:', sorted(roots) == sorted(errors.keys()))

# now the rescaled BM
cutoffs = [min(r, 35) for r in range(1, 71)]
bm = BMState(gf, cutoffs, n=70)
for r in range(1, 71):
    bm.round_fwd(r, S)
    if bm.ev_shift_top or bm.fail_cutoff or bm.fail_stack:
        print(f'round {r}: shift_top={bm.ev_shift_top} cutoff={bm.fail_cutoff} stack={bm.fail_stack}')
    if r <= 12 or r % 10 == 0:
        print(f'round {r}: d={bm.d} deg_sigma={_poly_deg(bm.sigma[:36])} tau0={bm.tau[0]} tau35={bm.tau[35]}')
sig = bm.sigma
print('rescaled final deg sigma:', _poly_deg(sig[:36]))
roots2 = []
beta = 1
for j in range(4095):
    if j: beta = gf.mul(beta, ginv)
    if poly_eval(sig[:36], beta, gf) == 0:
        roots2.append(j)
print('rescaled roots count:', len(roots2), 'match:', sorted(roots2) == sorted(errors.keys()))
print('rescaled roots[:8]:', sorted(roots2)[:8])
print('sigma0:', sig[0], 'd final:', bm.d)
EOF
