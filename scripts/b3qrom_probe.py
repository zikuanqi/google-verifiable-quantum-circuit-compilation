#!/usr/bin/env python
"""Probe: GF(2) rank of the baseline ANGLE delta difference banks (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_6_17493_sources/src/dfthcsos/run002")

import numpy as np

import banked_lookup as BL

npz = np.load("/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission/dfthcsos/submission.npz")
words = npz["angle_words"]
print("angle_words", words.shape, words.dtype, "max", words.max())
for J in (8, 10, 12):
    deltas = BL.batch_deltas(words, J)
    print(f"J={J}: K={len(deltas)} batches, delta shape {deltas[0].shape}")
    for mode in ("baseline", "zero", "linear"):
        rep = BL.difference_tables(deltas[0], brot=12, mode=mode)[-1]
        print(f"   mode={mode}: bank_width={rep['bank_width']}, gf2_rank={rep['gf2_rank']}, "
              f"active={rep['active_columns']}, basis_cnots={rep['basis_cnot_count']}")
    # rank across ALL batches
    ranks = [BL.difference_tables(d, brot=12, mode="linear")[-1]["gf2_rank"] for d in deltas]
    widths = [BL.difference_tables(d, brot=12, mode="linear")[-1]["bank_width"] for d in deltas]
    print(f"   all batches: max gf2_rank={max(ranks)}, max bank_width={max(widths)}")
