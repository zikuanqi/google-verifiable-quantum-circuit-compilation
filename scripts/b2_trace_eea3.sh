#!/bin/bash
# b2: trace the new EEA with the top-region zero test.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
sys.path.insert(0, 'src')
from b2_model import GF12, _align, _poly_deg, _poly_zero

gf = GF12()
E = [7, 0, 5, 3] + [0]*14
O = [9, 1, 2, 4] + [0]*14

r0 = _align(E); r1 = _align(O)
d0, d1 = _poly_deg(E), _poly_deg(O)
if d0 < d1:
    r0, r1 = r1, r0; d0, d1 = d1, d0
c = 0; done = _poly_zero(r1)
for it in range(12):
    q = gf.mul(r0[17], gf.inv(r1[17]))
    for i in range(18):
        r0[i] ^= gf.mul(q, r1[i])
    r0[17] ^= q
    r0 = [r0[-1]] + r0[:-1]
    c += 1
    zero = all(x == 0 for x in r0[18-d1:18]) if d1 > 0 else True
    fire = (not done and c >= d0 - d1 + 1 and (r0[17] != 0 or (d1 == 0 and zero)))
    print(f' it {it}: q={q} c={c} d0={d0} d1={d1} r0[17]={r0[17]} zero={zero} fire={fire}')
    if fire:
        r0, r1 = r1, r0
        d0, d1 = d1, d0 - c
        c = 0
        done = _poly_zero(r1)
        print(f'    SWAP -> d0,d1 = {d0},{d1} done={done} r1={r1}')
print('final r1_zero =', _poly_zero(r1))
EOF
