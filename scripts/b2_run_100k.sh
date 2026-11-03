#!/bin/bash
# b2: large failure-rate measurement (100k samples).
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import sys, time
sys.path.insert(0, 'src')
import b2_model as M

cutoffs = [min(r, 35) for r in range(1, 71)]
res = M.run_experiment(num_samples=100000, seed=7, cutoffs=cutoffs, K_cut=35)
for k, v in res.items():
    print(f'  {k}: {v}')
EOF
