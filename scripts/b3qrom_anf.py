#!/usr/bin/env python
"""Analyze the ANF structure of the baseline ANGLE delta columns (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_6_17493_sources/src/dfthcsos/run002")

import numpy as np

import banked_lookup as BL


def anf_count(f, n):
    """Number of nonzero ANF terms of a truth-table f over n bits (Mobius)."""
    t = f.copy()
    for i in range(n):
        t = t.reshape(-1, 2, 1 << (n - i - 1))
        t[:, 1, :] ^= t[:, 0, :]
        t = t.reshape(-1, 1 << (n - i - 1))
    return int(t.sum())


def anf_terms(f, n):
    """Set of product-support bitmasks in the ANF of f."""
    t = f.copy().reshape(1 << n)
    for i in range(n):
        half = 1 << (n - i - 1)
        t = t.reshape(-1, 2, half)
        t[:, 1, :] ^= t[:, 0, :]
        t = t.reshape(1 << n)
    masks = np.flatnonzero(t)
    return masks


def main():
    npz = np.load("/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission/dfthcsos/submission.npz")
    words = npz["angle_words"]
    deltas = BL.batch_deltas(words, 8)
    N = 12
    for k in (0, 1):
        delta = deltas[k]
        A, J = delta.shape
        W = J * 12
        # truth tables: 4096 entries; rows >= A are free (delta 0 on them)
        cols = []
        for j in range(J):
            for b in range(12):
                col = np.zeros(1 << N, dtype=np.uint8)
                col[:A] = (delta[:, j] >> b) & 1
                cols.append(col)
        # per-column degree and term counts
        all_terms = set()
        degree_stats = {}
        for c, col in enumerate(cols):
            masks = anf_terms(col, N)
            deg = max((int(m).bit_count() for m in masks), default=0)
            degree_stats[c] = (deg, len(masks))
            all_terms |= set(int(m) for m in masks)
        degs = [v[0] for v in degree_stats.values()]
        lens = [v[1] for v in degree_stats.values()]
        print(f"k={k}: W={W} columns, degrees: max={max(degs)} "
              f"hist={ {d: degs.count(d) for d in sorted(set(degs))} }")
        print(f"   per-column ANF terms: max={max(lens)} sum={sum(lens)}")
        print(f"   distinct shared product-supports: {len(all_terms)}")
        print(f"   shared supports by degree: { {d: sum(1 for m in all_terms if int(m).bit_count()==d) for d in range(6)} }")
        # number of distinct degree-2 supports used
        quad = [m for m in all_terms if int(m).bit_count() == 2]
        print(f"   distinct quadratic supports: {len(quad)} of 66")
        # are the columns affine over the 'pair products'? compute residual rank over GF(2)
        # build the matrix: rows = distinct pair supports used; columns = 4096 truth values
        if quad:
            P = np.zeros((len(quad), 1 << N), dtype=np.uint8)
            for i, m in enumerate(quad):
                idx = np.arange(1 << N)
                for bit in range(N):
                    if (m >> bit) & 1:
                        idx = np.where((idx >> bit) & 1 == 1, idx, -1)
                # compute the AND truth table properly
                tt = np.ones(1 << N, dtype=np.uint8)
                for bit in range(N):
                    if (m >> bit) & 1:
                        tt &= (np.arange(1 << N) >> bit) & 1
                P[i] = tt
            # for each column: solve col = affine + P^T c via gaussian elimination on rows
            # count columns expressible as affine+quadratic
            good = 0
            for col in cols:
                aug = np.vstack([np.ones(1 << N, dtype=np.uint8),
                                 np.array([(np.arange(1 << N) >> i) & 1 for i in range(N)], dtype=np.uint8),
                                 P]).astype(np.uint8)
                # solve aug^T c = col over GF(2)
                M = aug.T.astype(np.uint8)
                bvec = col.copy()
                # gaussian elimination
                m = M.copy()
                b = bvec.copy()
                rank = 0
                r = 0
                for c in range(m.shape[1]):
                    piv = -1
                    for rr in range(r, m.shape[0]):
                        if m[rr, c]:
                            piv = rr
                            break
                    if piv < 0:
                        continue
                    m[[r, piv]] = m[[piv, r]]
                    b[r], b[piv] = b[piv], b[r]
                    for rr in range(m.shape[0]):
                        if rr != r and m[rr, c]:
                            m[rr] ^= m[r]
                            b[rr] ^= b[r]
                    r += 1
                if np.any(b[r:]):
                    pass  # not expressible
                else:
                    good += 1
            print(f"   columns expressible as affine+quadratic over shared pairs: {good}/{W}")


if __name__ == "__main__":
    main()
