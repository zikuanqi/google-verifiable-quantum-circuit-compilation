#!/bin/bash
# b2: model-level round structure of the 8 findjunk4 shots; correlate with
# the observed phase flips [0,0,1,1,1,0,1,0].
set -e
export PYTHONPATH="/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix:/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src"
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np

m, n, degree = 15, 6, 4
ORDER = 15
log = [-1] * 16
antilog = [0] * 16
x = 1
for i in range(ORDER):
    log[x] = i
    antilog[i] = x
    x <<= 1
    if x & 0x10:
        x ^= 0x13  # x^4+x+1
MUL = [[0]*16 for _ in range(16)]
for a in range(16):
    for b in range(16):
        MUL[a][b] = 0 if (a == 0 or b == 0) else antilog[(log[a]+log[b]) % ORDER]
def inv(a):
    if a == 0: return 0
    v = 1
    for _ in range(14):
        v = MUL[v][a]
    return v

synd = np.random.default_rng(1).integers(0, 1 << degree, size=(8, n), dtype=np.int64)
K_r = [min(r, n // 2) for r in range(1, n + 1)]
print('observed flips:', [0, 0, 1, 1, 1, 0, 1, 0])
for s in range(8):
    S = [int(v) for v in synd[s]]
    sigma = [0]*4; tau = [0]*4
    sigma[0] = 1; tau[0] = 1
    d = 0
    stack = []
    rec = []  # per round: (pushed, rep, delta)
    for r in range(1, n + 1):
        Km = min(K_r[r-1], 3)
        K = min(Km, r-1)
        old = tau[3]
        tau = [old] + tau[:3]
        delta = 0
        for i in range(K+1):
            delta ^= MUL[sigma[i]][S[r-1-i]]
        nz = delta != 0
        rep = nz and d >= 0
        pushed = d < 0
        alpha = MUL[delta][sigma[0]]
        if rep:
            sigma, tau = tau[:], sigma[:]
        mi = inv(alpha) if rep else alpha
        for i in range(Km+1):
            sigma[i] ^= MUL[mi][tau[i]]
        if pushed:
            stack.append((r, delta))
        rec.append((pushed, rep, delta, d))
        d = (-d if rep else d + 1)
    # reverse pass
    pat = []
    pops = 0
    for r in range(n, 0, -1):
        pushed, rep, delta, d_pre = rec[r-1]
        d_c = d - 1
        entry = stack[-1] if stack else None
        rt = entry is not None and entry[0] == r
        rep_r = (d_c < 0) and not rt
        pat.append('R' if rep_r else ('P' if rt else '.'))
        if rt:
            stack.pop()
            pops += 1
        # restore d
        d = (-d_c - 1 if rep_r else d_c)
        # undo swap+madd+shift (state-level, to keep d/sigma correct)
        Km = min(K_r[r-1], 3)
        K = min(Km, r-1)
        # reverse madd
        if rep_r:
            m = MUL[sigma[0]][inv(tau[0])]
        else:
            sdelta = entry[1] if rt else 0
            m = MUL[sdelta][sigma[0]]
        if rep_r or rt:
            for i in range(Km+1):
                sigma[i] ^= MUL[m][tau[i]]
            if rep_r:
                sigma, tau = tau[:], sigma[:]
        # unshift tau
        old0 = tau[0]
        tau = tau[1:] + [old0]
        # d restored
    print(f'shot{s}: rounds={"".join(pat)} rep={sum(1 for c in pat if c=="R")} pop={pops}')
EOF