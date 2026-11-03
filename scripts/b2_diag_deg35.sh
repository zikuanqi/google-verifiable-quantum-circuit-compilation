#!/bin/bash
# b2: instrument the deg-35 reverse failures.
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
rng = np.random.default_rng(1)
cutoffs = [min(r, 35) for r in range(1, 71)]
shown = 0
for trial in range(2000):
    s = dist.sample(rng)
    S = s['syndromes'][:]
    bm = M.BMState(gf, cutoffs, n=70)
    for r in range(1, 71):
        bm.round_fwd(r, S)
    if s['weight'] != 35:
        continue
    # trace the reverse on this deg-35 sample
    sigma = [x for x in bm.sigma]
    tau0_snapshot = [x for x in bm.tau]
    d0_snapshot = bm.d
    st0 = [x for x in bm.stack]
    # reverse and detect the first round where the rep/rt flags mismatch
    # (recompute the true flags from the forward trace)
    # Re-run forward recording (rep, ret, delta, m) per round
    bm2 = M.BMState(gf, cutoffs, n=70)
    rec = []
    for r in range(1, 71):
        d, rep, ret = bm2.round_fwd(r, S)
        rec.append((rep, ret, d))
    # reverse with diagnostics
    bm3 = M.BMState(gf, cutoffs, n=70)
    for r in range(1, 71):
        bm3.round_fwd(r, S)
    for r in range(70, 0, -1):
        rep_true, ret_true, d_true = rec[r-1]
        rep_det = bm3.tau[0] != 0 and bm3.d <= 0 and bm3.tau[35] == 0
        rt_det = (bm3.stack[-1][0] == r) if bm3.stack else False
        if rep_det != rep_true or rt_det != ret_true:
            print(f'trial {trial} round {r}: true(rep={rep_true},ret={ret_true}) '
                  f'det(rep={rep_det},rt={rt_det}) d={bm3.d} tau0={bm3.tau[0]} '
                  f'tau35={bm3.tau[35]} sigma35={bm3.sigma[35]}')
            shown += 1
            break
        bm3.round_rev(r, S)
    if shown >= 6:
        break
print('shown:', shown)
EOF
