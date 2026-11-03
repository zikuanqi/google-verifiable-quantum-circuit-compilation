#!/bin/bash
# b2: trace the uneliminate on a tiny case.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys
sys.path.insert(0, 'src')
import b2_model as M
gf = M.GF12()
E = [7, 0, 5, 3] + [0]*14
O = [9, 1, 2, 4] + [0]*14
r0 = M._align(E); r1 = M._align(O)
d0, d1 = M._poly_deg(E), M._poly_deg(O)
r0i, r1i = [x for x in r0], [x for x in r1]
r0f, r1f, qs, fires, d0f, d1f, a_f = M.eea_eliminate(r0, r1, d0, d1, gf)
r0u, r1u = M.eea_uneliminate(r0f, r1f, qs, fires, gf)
print('fires:', fires)
print('restore r0 ok:', r0u == r0i, ' r1 ok:', r1u == r1i)
print('r0i:', r0i)
print('r0u:', r0u)
print('r1i:', r1i)
print('r1u:', r1u)
EOF
