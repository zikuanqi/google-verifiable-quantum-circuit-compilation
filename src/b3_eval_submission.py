# -*- coding: utf-8 -*-
"""Evaluate an external/public submission npz with the official verifier math."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="submission.npz path")
    args = ap.parse_args()

    ref_data = ref.load_reference(ref.check_fcidump())
    fact = np.load(args.submission, allow_pickle=False)
    params = dict(fact)
    params.update(ref.load_fixed_shifts())
    w = np.asarray(params["W"], dtype=np.float64)
    W = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    S = np.asarray(dm.symmetric_from_lower(params["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(S))
    U = np.array(fact["U"], dtype=np.float64)
    print(f"shapes: U {U.shape}  w {w.shape}  H2_I {fact['H2_I'].shape}")

    words = np.array(fact["angle_words"]) if "angle_words" in fact.files else None
    tree = np.array(fact["tree_edges"])
    if words is not None:
        wmax = int(words.max())
        brot = max(1, wmax.bit_length())
        print(f"angle_words max {wmax} -> brot {brot}")
        m = dm.evaluate_dfthc_angle_words(
            words, w, W, S, beta_1, ref_data, brot=brot, tree_edges=tree,
            energy_upper=ref.UPPER_ENERGY)
        print("== quantized ==")
        for k in ("error_1", "error_2", "Lambda", "E_SOS", "E_gap", "lambda_eff"):
            print(f"  {k:>10s} = {m[k]:.10g}")
        print(f"  gap_in_domain = {m['gap_in_domain']}")

    X = dm.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_prime = dm.build_h1_prime(ref_data["h1_thc"], S, beta_1,
                                 ref_data["electrons"], W, X)
    _, evecs = np.linalg.eigh(h1_prime)
    u_all = np.vstack([evecs.T, U.reshape(-1, U.shape[-1])])
    mu = dm.evaluate_dfthc_vectors(u_all, w, W, S, beta_1, ref_data,
                                   energy_upper=ref.UPPER_ENERGY)
    print("== unquantized (float U) ==")
    for k in ("error_1", "error_2", "Lambda", "E_SOS", "E_gap", "lambda_eff"):
        print(f"  {k:>10s} = {mu[k]:.10g}")


if __name__ == "__main__":
    main()
