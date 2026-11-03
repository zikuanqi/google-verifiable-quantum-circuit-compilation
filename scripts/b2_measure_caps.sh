#!/bin/bash
# b2: measure the reference-scheme stack depth / overflow vs CAPS.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys, time
sys.path.insert(0, 'src')
import b2_model as M

cutoffs = [min(r, 35) for r in range(1, 71)]
num = 20000
import numpy as np
dist = M.DQIDistribution(gf=M.GF12())
rng = np.random.default_rng(3)
samples = [dist.sample(rng) for _ in range(num)]
print('samples ready')

for caps in (1, 2, 3, 4):
    n_fail = 0
    n_other = 0
    maxdepth = 0
    depths = [0] * 8
    for s in samples:
        bm = M.BMState(M.GF12(), cutoffs, n=70, caps=caps)
        for r in range(1, 71):
            bm.round_fwd(r, s['syndromes'])
        maxdepth = max(maxdepth, len(bm.stack))
        if len(bm.stack) < len(depths):
            depths[len(bm.stack)] += 1
        else:
            depths[-1] += 1
        if bm.fail_stack:
            n_fail += 1
            continue
        # reverse
        for r in range(70, 0, -1):
            bm.round_rev(r, s['syndromes'])
        ok = (bm.sigma == [1] + [0] * 35 and bm.tau == [1] + [0] * 35
              and bm.d == 0 and not bm.stack and not bm.fail_cutoff)
        if not ok:
            n_other += 1
    print(f'caps={caps}: fail_stack={n_fail} ({n_fail/num:.2e}) '
          f'rev_mismatch={n_other} maxdepth={maxdepth} final_depth_hist={depths}')
EOF
