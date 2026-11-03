#!/bin/bash
# b2: definitive EEA test vs textbook reference.
set -e
export PYTHONPATH=/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix
PY=/home/kaggle/miniconda3/envs/py313/bin/python
cd /mnt/d/kaggle/google-verifiable-quantum-circuit-compilation
$PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'src')
import b2_model as M

gf = M.GF12()

def pmul(P, Q, n):
    R = [0]*n
    for i, p in enumerate(P):
        if p:
            for j, q in enumerate(Q):
                if q and i+j < n:
                    R[i+j] ^= gf.mul(p, q)
    return R

def pdivmod(P, Q):
    P = P[:]; dP = M._poly_deg(P); dQ = M._poly_deg(Q)
    if dQ < 0: return None
    lcQ = Q[dQ]
    for i in range(dP, dQ-1, -1):
        if P[i]:
            coef = gf.mul(P[i], gf.inv(lcQ))
            for j in range(dQ+1):
                P[i-dQ+j] ^= gf.mul(coef, Q[j])
    while P and P[-1] == 0: P.pop()
    return P

def textbook_eea(E, O):
    r0 = [x for x in E if x != 0] or [0]
    r1 = [x for x in O if x != 0] or [0]
    s0, t0 = [1], [0]
    s1, t1 = [0], [1]
    while M._poly_deg(r1) >= 0:
        r0 = [x for x in r0]
        q = []
        dP, dQ = M._poly_deg(r0), M._poly_deg(r1)
        if dP >= dQ:
            q = [0]*(dP-dQ+1)
            lcQ = r1[dQ]
            while M._poly_deg(r0) >= dQ:
                dP = M._poly_deg(r0)
                coef = gf.mul(r0[dP], gf.inv(lcQ))
                q[dP-dQ] = coef
                for j in range(dQ+1):
                    r0[dP-dQ+j] ^= gf.mul(coef, r1[j])
                while r0 and r0[-1] == 0: r0.pop()
                r0 = r0 + [0]*(max(0, dQ) - len(r0) + 1)
                r0 = r0[:max(len(r0), dQ)]
        # full quotient poly mult: s0 -= q*s1 ; t0 -= q*t1
        qq = q + [0]*(1)
        s0n = pmul(qq, s1, max(len(s0), len(s1)+len(q)))[:]
        for i, v in enumerate(s0):
            if i < len(s0n): s0n[i] ^= v
        t0n = pmul(qq, t1, max(len(t0), len(t1)+len(q)))[:]
        for i, v in enumerate(t0):
            if i < len(t0n): t0n[i] ^= v
        r0, r1 = r1, r0
        s0, s1 = s1, s0n
        t0, t1 = t1, t0n
        # trim
        while len(r0) > 1 and r0[-1] == 0: r0.pop()
        while len(r1) > 1 and r1[-1] == 0: r1.pop()
    return s0, t0, r0[0]

rng = np.random.default_rng(7)
dE, dO = 14, 13
E = [0]*18; O = [0]*18
E[0] = int(rng.integers(1, 4096))
for i in range(1, dE+1): E[i] = int(rng.integers(0, 4096))
for i in range(1, dO+1): O[i] = int(rng.integers(0, 4096))
O[0] = int(rng.integers(1, 4096))
A_ref, B_ref, g_ref = textbook_eea(E, O)
print('textbook: degA:', M._poly_deg(A_ref), 'degB:', M._poly_deg(B_ref), 'gcd:', g_ref)

# tracking
r0 = M._align(E); r1 = M._align(O)
d0, d1 = dE, dO
r0f, r1f, s0f, s1f, t0f, t1f, d0f, d1f, a_f = M.eea_eliminate(r0, r1, d0, d1, gf)
a = a_f % 19
Aw = [s0f[(a+i) % 19] for i in range(19)]
Bw = [t0f[(a+i) % 19] for i in range(19)]
cg = r0f[a_f % 18]
print('tracking: degA:', M._poly_deg(Aw), 'degB:', M._poly_deg(Bw), 'cg:', cg)
# compare the low parts: A_tracking == A_ref * g_ref/cg?  (scale: A*E+B*O=cg vs gcd g_ref)
prodA = [0]*40
for i in range(19):
    for j in range(18):
        prodA[i+j] ^= gf.mul(Aw[i], E[j])
prodB = [0]*40
for i in range(19):
    for j in range(18):
        prodB[i+j] ^= gf.mul(Bw[i], O[j])
prod = [x ^ y for x, y in zip(prodA, prodB)]
nz = [i for i, x in enumerate(prod) if x]
print('tracking bezout nz:', nz[:8], 'count:', len(nz))
print('A_ref[:8]:', A_ref[:8])
print('Aw[:8]:', Aw[:8])
# scale factor: tracking (A,B) should = (A_ref, B_ref) * (cg/g_ref)
sc = gf.mul(cg, gf.inv(g_ref))
print('scaled A_ref:', [gf.mul(x, sc) for x in A_ref[:8]])
EOF
