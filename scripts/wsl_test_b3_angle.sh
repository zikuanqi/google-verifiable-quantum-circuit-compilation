#!/bin/bash
# Test the kernel-side B3 angle builder: rebuild the baseline's angle circuits
# from sample_submission/dfthcsos/submission.npz and verify with the official
# verifier (expect score 3.3556e9).
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
ROOT=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
OUT=$ROOT/results/b3_angle_rebuild
rm -rf "$OUT"; mkdir -p "$OUT"
$PY - <<'EOF'
import numpy as np
import b3_angle_build
fact = dict(np.load("/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission/dfthcsos/submission.npz", allow_pickle=False))
# baseline: brot=12, J=8; the npz has angle_words (3976,75)
out = "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b3_angle_rebuild"
import os
brot, J = 12, 8
words = np.asarray(fact["angle_words"])
deltas = b3_angle_build.batch_deltas(words, J)
print("num batches:", len(deltas))
for k, delta in enumerate(deltas):
    c = b3_angle_build.build_angle_k_circuit(delta, brot)
    c.to_file(os.path.join(out, f"angle_k_{k}.kmb"), format="kmb")
    st = c.operation_statistics()
    print(f"  angle_k_{k}: {st.num_toffoli_like} T, {c.num_qubits} q")
EOF
# copy the factorization npz (without angle_words) for the verifier
$PY - <<'EOF'
import numpy as np
out = "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b3_angle_rebuild"
d = np.load("/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission/dfthcsos/submission.npz", allow_pickle=False)
np.savez(f"{out}/submission.npz", **{k: d[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower", "tree_edges")})
EOF
cd $ROOT/data/verifiers
$PY dfthcsos/verify.py "$OUT" 2>&1 | $PY -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ('status','score','lambda_eff','toffoli_count','num_qubits') if k in d}, indent=2))"
