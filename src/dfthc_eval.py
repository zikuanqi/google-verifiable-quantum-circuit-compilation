# -*- coding: utf-8 -*-
"""Benchmark III (DFTHC-SOS) classical metrics evaluator — pure numpy, no kickmix.

Uses the official verifier's own math (`dfthcsos.scoreinfra.dfthc_math` and
`reference`) to reproduce the acceptance metrics from a factorization + angle
words, so results match the Kaggle grader exactly.

Usage:
    python src/dfthc_eval.py [--factorization PATH] [--angle-words PATH] [--brot N]
"""
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


def load_baseline():
    data = ROOT / "data" / "sample_submission" / "dfthcsos"
    fact = np.load(data / "submission.npz", allow_pickle=False)
    return data, fact


def native_inputs(params):
    """Replicate dfthcsos.verify.native_inputs (kickmix-free)."""
    w = np.asarray(params["W"], dtype=np.float64)                 # lowercase w: (R, B, C)
    W = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)  # identity offsets: (R, C)
    S = np.asarray(
        dm.symmetric_from_lower(params["Hsym_lower"], np.shape(params["U"])[-1]),
        dtype=np.float64,
    )
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(S))
    return w, W, S, beta_1


def evaluate(factorization: Path, angle_words: np.ndarray, brot: int):
    ref_data = ref.load_reference(ref.check_fcidump())
    fact = np.load(factorization, allow_pickle=False)
    params = dict(fact)
    # The verifier overrides the submitted shift with the packaged baseline shift.
    params.update(ref.load_fixed_shifts())
    w, W, S, beta_1 = native_inputs(params)
    U = np.array(fact["U"])

    tree_edges = np.array(fact["tree_edges"])
    metrics = dm.evaluate_dfthc_angle_words(
        angle_words, w, W, S, beta_1, ref_data,
        brot=brot, tree_edges=tree_edges, energy_upper=ref.UPPER_ENERGY,
    )

    # Unquantized comparison: exact h1_prime eigenvectors as one-body rows
    # plus the float U vectors as two-body rows.
    X = dm.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_prime = dm.build_h1_prime(
        ref_data["h1_thc"], S, beta_1, ref_data["electrons"], W, X
    )
    _, evecs = np.linalg.eigh(h1_prime)
    u_all = np.vstack([evecs.T, U.reshape(-1, U.shape[-1])])
    metrics_unquant = dm.evaluate_dfthc_vectors(
        u_all, w, W, S, beta_1, ref_data,
        energy_upper=ref.UPPER_ENERGY,
    )
    return metrics, metrics_unquant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factorization", default=None)
    ap.add_argument("--angle-words", default=None)
    ap.add_argument("--brot", type=int, default=12)
    args = ap.parse_args()

    data, fact = load_baseline()
    fact_path = args.factorization or data / "submission.npz"
    if args.angle_words:
        words = np.load(args.angle_words, allow_pickle=False)["angle_words"]
    else:
        fact_npz = np.load(fact_path, allow_pickle=False)
        if "angle_words" in fact_npz.files:
            words = np.array(fact_npz["angle_words"])
        else:
            words = np.array(fact["angle_words"])

    m, mu = evaluate(Path(fact_path), words, args.brot)
    print("== quantized (angle words, brot=%d) ==" % args.brot)
    for k in ("error_1", "error_2", "Lambda", "E_SOS", "E_gap", "lambda_eff"):
        print(f"  {k:>10s} = {m[k]:.10g}")
    print(f"  gap_in_domain = {m['gap_in_domain']}")
    print("== unquantized (float U) ==")
    for k in ("error_1", "error_2", "Lambda", "E_SOS", "E_gap", "lambda_eff"):
        print(f"  {k:>10s} = {mu[k]:.10g}")


if __name__ == "__main__":
    main()
