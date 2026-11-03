#!/usr/bin/env python
"""Recover the 19.5 team's ANGLE delta tables by simulation and ANF-analyze them (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/verifiers")

import numpy as np
import kickmix as km
from dfthcsos.scoreinfra.circuits import register_layout

P = "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_19_53750_public_resources/run012a_public_release/payload/dfthcsos"
npz = np.load(P + "/submission.npz")
print("npz keys:", npz.files)
U = npz["U"]
print("U shape:", U.shape)
R, B, C = U.shape[0], U.shape[1], npz["W"].shape[2]
A = 76 + R * B
print("A =", A, "selection bits:", (A - 1).bit_length())

c0 = km.Circuit.from_file(P + "/angle_k_0.kmb")
c1 = km.Circuit.from_file(P + "/angle_k_1.kmb")
lay = register_layout([c0, c1])
nsel = len(lay.selection)
J = len(lay.words)
brot = len(lay.words[0])
print("J =", J, "brot =", brot)

addrs = np.arange(A, dtype=np.int64)
abits = ((addrs[None, :] >> np.arange(nsel, dtype=np.int64)[:, None]) & 1).astype(bool)

def run(circ):
    sim = km.Simulator(batch_size=A, debug_print=False, seed=1, counting_operations=True)
    sim.use_same_registers_as(circ)
    sim.clear_for_shot()
    sim.write_across_shots(lay.selection, abits)
    sim.do(circ)
    out = []
    for w in lay.words:
        bits = sim.read_across_shots(w)
        vals = (bits.T.astype(np.int64) * (1 << np.arange(brot))).sum(axis=1)
        out.append(vals)
    return np.stack(out, axis=1)

b0 = run(c0)
b1 = run(c1)
delta1 = b1 ^ b0
print("delta1 shape:", delta1.shape, "max:", delta1.max())

# ANF analysis of delta1 columns
N = nsel
cols = []
for j in range(J):
    for b in range(brot):
        col = np.zeros(1 << N, dtype=np.uint8)
        col[:A] = (delta1[:, j] >> b) & 1
        cols.append(col)

def anf_terms(f, n):
    t = f.copy().reshape(1 << n)
    for i in range(n):
        half = 1 << (n - i - 1)
        t = t.reshape(-1, 2, half)
        t[:, 1, :] ^= t[:, 0, :]
        t = t.reshape(1 << n)
    return np.flatnonzero(t)

all_terms = set()
max_terms = 0
for col in cols:
    masks = anf_terms(col, N)
    max_terms = max(max_terms, len(masks))
    all_terms |= set(int(m) for m in masks)
print("19.5 team delta1: per-column ANF terms max:", max_terms)
print("distinct shared supports:", len(all_terms), "of", 1 << N)
