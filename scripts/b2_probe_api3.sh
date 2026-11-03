#!/bin/bash
# b2: dump kickmix method NAMES only (short).
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
$PY - <<'EOF'
import kickmix as km
for cls_name in ('CircuitBuilder', 'Circuit', 'Simulator', 'GF2Field', 'OperationStatistics'):
    cls = getattr(km, cls_name)
    print(f'==== {cls_name} ====')
    for n in dir(cls):
        if not n.startswith('_'):
            print('  ', n)
    print()
EOF
