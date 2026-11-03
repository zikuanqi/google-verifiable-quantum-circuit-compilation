# -*- coding: utf-8 -*-
"""Quick evaluation of b3 candidate npz files with official math."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "verifiers"))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref


def eval_candidate(path: Path):
    d = np.load(path)
    U = d["U"] if "U" in d.files else None
    w = d["w"] if "w" in d.files else d["W"]
    print(f"--- {path.name}: U={None if U is None else U.shape} w={w.shape}")
    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = dm.symmetric_from_lower(shifts["Hsym_lower"], 76)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    X = dm.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h2_delta = dm.reconstruct_h2(X) - ref_data["tei"]
    diag = np.arange(76)
    h2_delta[:, :, diag, diag] -= 0.5 * S[:, :, None]
    h2_delta[diag, diag, :, :] -= 0.5 * S[None, :, :]
    e2 = float(np.linalg.norm(h2_delta.ravel()))
    # W: zero offset for now; check lambda_2
    R, B, C = w.shape
    W = np.zeros((R, C))
    lam2 = dm.lambda_h2(w, W)
    print(f"  error_2 = {e2:.6f}  Lambda2(w, W=0) = {lam2:.3f}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        eval_candidate(Path(name))
