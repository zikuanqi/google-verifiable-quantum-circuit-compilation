# -*- coding: utf-8 -*-
"""Export the baseline30 init as a trainer-format warm-start npz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
RESULTS = ROOT / "results"
sys.path.insert(0, str(VERIFIERS))
sys.path.insert(0, str(ROOT / "src"))

from dfthcsos.scoreinfra import reference as ref  # noqa: E402
import b3_optimize as bo  # noqa: E402


def main() -> None:
    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    prob = bo.Problem()
    U, w, W = bo.init_baseline30(prob, 65, 60, 30)
    H2_I = W + w.sum(axis=1)
    np.savez(
        RESULTS / "b3_warm_65_60_30_b30.npz",
        U=U.astype(np.float32),
        W=w.astype(np.float32),
        H2_I=H2_I.astype(np.float32),
        H1_I=shifts["H1_I"].astype(np.float32),
        Hsym_lower=shifts["Hsym_lower"].astype(np.float32),
    )
    print("wrote results/b3_warm_65_60_30_b30.npz")


if __name__ == "__main__":
    main()
