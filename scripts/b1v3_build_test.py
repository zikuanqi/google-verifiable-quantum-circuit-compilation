#!/usr/bin/env python
"""Build the B1 v3 circuit from a coefficient .npz and print statistics (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import numpy as np

import b1v3_build as b3b
import b1v3_model as v3m


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/community/toolbox/"
        "fusion/selected_coefficients.npz")
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1v3_test.kmb")
    npz = np.load(src)
    tab = npz["table"]
    cfg = v3m.V3Cfg(f2=int(npz["config"][0]), f1=int(npz["config"][1]),
                    fyr=int(npz["config"][2]), gy=int(npz["config"][3]),
                    drop2=tuple(int(v) for v in npz["drop"]))
    tabs = {(e, a): tuple(int(v) for v in tab[a + 128 * e]) for e in (0, 1) for a in range(128)}
    lay = v3m.make_layout(cfg, tabs)
    c = b3b.build_v3_circuit(cfg, tabs, lay)
    st = c.operation_statistics()
    print(f"qubits={c.num_qubits} toffoli={st.num_toffoli_like} t={st.num_t_like} "
          f"rot={st.num_z_pow(min_clifford_hierarchy_level=4)} ops={len(c)}")
    c.to_file(out, format="kmb")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
