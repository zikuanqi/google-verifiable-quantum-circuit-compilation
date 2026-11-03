#!/bin/bash
# b2: dump kickmix method signatures + docstrings.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import kickmix as km

def dump(cls_name):
    cls = getattr(km, cls_name)
    print(f'======== {cls_name} ========')
    for n in dir(cls):
        if n.startswith('_'):
            continue
        try:
            obj = getattr(cls, n)
        except Exception as e:
            print(f'  {n}: <error {e}>')
            continue
        if callable(obj):
            doc = (getattr(obj, '__doc__', None) or '').strip()
            try:
                sig = str(getattr(obj, '__text_signature__', '')) or ''
            except Exception:
                sig = ''
            line = f'  {n}{sig}'
            print(line)
            if doc:
                for dl in doc.splitlines():
                    print(f'      | {dl}')
        else:
            print(f'  {n} = {obj!r}')
    print()

for c in ('CircuitBuilder', 'Circuit', 'Simulator', 'GF2Field', 'OperationStatistics'):
    dump(c)
EOF
