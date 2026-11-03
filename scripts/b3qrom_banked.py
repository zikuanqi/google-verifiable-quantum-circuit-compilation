#!/usr/bin/env python
"""Measure the banked 2-address QROM for the baseline deltas at several J (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/qrom_bmpsi/vqcc_6_17493_sources/src/dfthcsos/run002")

import numpy as np

import banked_lookup as BL


def main():
    npz = np.load("/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/sample_submission/dfthcsos/submission.npz")
    words = npz["angle_words"]
    for J in (8, 5, 4):
        deltas = BL.batch_deltas(words, J)
        report = BL.difference_tables(deltas[0], brot=12, mode="linear")[-1]
        print(f"J={J}: bank_width={report['bank_width']}, rank={report['gf2_rank']}, "
              f"cnots={report['basis_cnot_count']}")
        try:
            c, rep = BL.build_qroam2(deltas[0], brot=12, mode="linear", return_report=True)
            print(f"   circuit: qubits={rep['num_qubits']}, static_toffoli={rep['static_toffoli']}")
        except Exception as ex:
            print(f"   build failed: {str(ex)[:100]}")


if __name__ == "__main__":
    main()
