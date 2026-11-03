#!/bin/bash
# b2: trace the EEA eliminate on a small case.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
sys.path.insert(0, 'src')
from b2_model import GF12, _align, _poly_deg, _poly_zero

gf = GF12()
E = [7, 0, 5, 3] + [0]*14    # deg 3
O = [9, 1, 2, 4] + [0]*14    # deg 3

def trace(r0, r1, d0, d1, max_iters=35):
    c = 0; done = _poly_zero(r1)
    qs = []
    for it in range(max_iters):
        q = gf.mul(r0[17], gf.inv(r1[17]))
        qs.append(q)
        for i in range(18):
            r0[i] ^= gf.mul(q, r1[i])
        r0[17] ^= q
        r0 = [r0[-1]] + r0[:-1]
        c += 1
        fire = (not done and c >= d0 - d1 + 1 and (r0[17] != 0 or (d1 == 0 and _poly_zero(r0))))
        if it < 14:
            print(f'  it {it}: q={q} c={c} d0={d0} d1={d1} r0[17]={r0[17]} fire={fire} zero={_poly_zero(r0)}')
        if fire:
            r0, r1 = r1, r0
            d0, d1 = d1, d0 - c
            c = 0
            done = _poly_zero(r1)
    return r0, r1, d0, d1

r0 = _align(E); r1 = _align(O)
d0, d1 = _poly_deg(E), _poly_deg(O)
if d0 < d1:
    r0, r1 = r1, r0; d0, d1 = d1, d0
print('start d0,d1 =', d0, d1)
r0f, r1f, d0f, d1f = trace(r0, r1, d0, d1)
print('final: d0,d1 =', d0f, d1f, 'r1_zero=', _poly_zero(r1f), 'r0=', r0f)
EOF
