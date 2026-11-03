#!/bin/bash
# b2: probe kickmix API surface.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import kickmix as km
print('module file:', km.__file__)
names = [n for n in dir(km) if not n.startswith('_')]
print('NAMES:', ' '.join(names))
print()
for name in ('CircuitBuilder', 'Circuit', 'Simulator', 'GF2Field', 'UnaryIterationCursor', 'array', 'q', 'b'):
    obj = getattr(km, name, None)
    print(f'--- {name}: {type(obj)}')
    if obj is not None:
        doc = (getattr(obj, '__doc__', None) or '').strip()
        print(doc[:1200])
        print()
EOF
