#!/bin/bash
# b2: instrument the model failures.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
from b2_model import (GF12, DQIDistribution, BMState, omega_product,
                      sigma_to_eo, eea_eliminate, eea_reconstruct,
                      eval_sigma_from_eo, eval_poly, eval_psi_from_ab,
                      _poly_deg, _align, PRIMITIVE)

gf = GF12()
dist = DQIDistribution(gf=gf)
rng = np.random.default_rng(3)
cutoffs = [min(r, 35) for r in range(1, 71)]

for trial in range(6):
    s = dist.sample(rng)
    S = s['syndromes'][:]
    errs = s['errors']
    bm = BMState(gf, cutoffs, n=70)
    for r in range(1, 71):
        bm.round_fwd(r, S)
    sig = bm.sigma
    # roots of sigma
    ginv = gf.inv(2); beta = 1; roots = []
    for j in range(4095):
        if j: beta = gf.mul(beta, ginv)
        if eval_poly(sig[:36], beta, gf) == 0:
            roots.append(j)
    roots_ok = sorted(roots) == sorted(errs.keys())
    print(f'trial {trial}: w={s["weight"]} deg_sig={_poly_deg(sig[:36])} roots_ok={roots_ok} '
          f'd={bm.d} stack={bm.stack} cutoff={bm.fail_cutoff} shifttop={bm.ev_shift_top}')
    if not roots_ok:
        print('   sigma[:6]=', sig[:6], ' sigma[30:36]=', sig[30:36])
        # trace a few rounds
        bm2 = BMState(gf, cutoffs, n=70)
        for r in range(1, 71):
            dd, rep, ret = bm2.round_fwd(r, S)
            if r in (2, 4, 10, 20, 40, 60):
                print(f'   r={r}: d={bm2.d} deg={_poly_deg(bm2.sigma[:36])} rep={rep} ret={ret} '
                      f'D={dd != 0}')
        break
EOF
