#!/bin/bash
# b2: diagnose the cutoff_fail=238 and verify the forward BM on many samples.
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
rng = np.random.default_rng(2)
cutoffs = [min(r, 35) for r in range(1, 71)]

n_cut = 0; n_roots_bad = 0; n_deg_bad = 0
shown = 0
for trial in range(2000):
    s = dist.sample(rng)
    S = s['syndromes'][:]
    bm = M.BMState(gf, cutoffs, n=70)
    for r in range(1, 71):
        bm.round_fwd(r, S)
        # degree bound check
        d = M._poly_deg(bm.sigma[:36])
        dt = M._poly_deg(bm.tau[:36])
        if d > r - 1 or dt > r - 1:
            n_deg_bad += 1
            if shown < 3:
                print(f'trial {trial} round {r}: deg sigma={d} deg tau={dt} (bound r-1={r-1})')
                shown += 1
    if bm.fail_cutoff:
        n_cut += 1
    # roots check
    ginv = gf.inv(2); beta = 1; roots = []
    for j in range(4095):
        if j: beta = gf.mul(beta, ginv)
        if M.eval_poly(bm.sigma[:36], beta, gf) == 0:
            roots.append(j)
    if sorted(roots) != sorted(s['errors'].keys()):
        n_roots_bad += 1
print(f'cutoff events: {n_cut}/2000')
print(f'degree-bound violations: {n_deg_bad}')
print(f'roots mismatches: {n_roots_bad}')
# weight distribution
w = dist.weight_probs
print('P(w=33):', w[33], 'P(w=34):', w[34], 'P(w=35):', w[35], 'P(w>=34):', w[34:].sum())
EOF
