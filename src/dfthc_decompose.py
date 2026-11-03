# -*- coding: utf-8 -*-
"""Decompose the baseline lambda_eff: Lambda1 vs Lambda2, E_SOS terms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref


def main():
    ref_data = ref.load_reference(ref.check_fcidump())
    fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                   allow_pickle=False)
    params = dict(fact)
    params.update(ref.load_fixed_shifts())
    w = np.asarray(params["W"], dtype=np.float64)
    W = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    S = np.asarray(dm.symmetric_from_lower(params["Hsym_lower"], 76), dtype=np.float64)
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(S))
    U = np.array(fact["U"])

    X = dm.reconstruct_x(U, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1_prime = dm.build_h1_prime(ref_data["h1_thc"], S, beta_1,
                                 ref_data["electrons"], W, X)
    evals = np.linalg.eigvalsh(h1_prime)
    lam1 = float(np.sum(np.abs(evals)))
    tr = float(np.sum(evals))
    lam2 = dm.lambda_h2(w, W)
    Lambda = lam1 + lam2
    print(f"Lambda1 (one-body ||h1'||_* ) = {lam1:.6f}")
    print(f"Lambda2 (two-body LCU)        = {lam2:.6f}")
    print(f"Lambda total                  = {Lambda:.6f}")
    print(f"  |evals| breakdown: pos={np.sum(evals[evals>0]):.4f}  neg={-np.sum(evals[evals<0]):.4f}")
    neg2 = float(np.sum(evals[evals < 0]))
    print(f"  (sum|e| - sum e) = 2*|neg sum| = {-2*neg2:.6f}")
    w2 = float(np.sum(W**2))
    print(f"  1/2 sum W^2                  = {0.5*w2:.6f}")
    ecore = ref_data["ecore"]
    print(f"  ecore = {ecore:.6f},  eta*beta_1 = {ref_data['electrons']*beta_1:.6f}")
    E_SOS = ecore - ref_data["electrons"]*beta_1 - (lam1 - tr) - 0.5*w2
    print(f"  E_SOS = {E_SOS:.6f}")
    print(f"  E_gap = {ref.UPPER_ENERGY - E_SOS:.6f}  (floor 3.914898)")
    lam_eff = np.sqrt((ref.UPPER_ENERGY - E_SOS) * (2*Lambda - (ref.UPPER_ENERGY - E_SOS)))
    print(f"  lambda_eff = {lam_eff:.6f}")
    # sensitivity: d lambda_eff / d Lambda at fixed E_gap
    E_gap = ref.UPPER_ENERGY - E_SOS
    print(f"  d(lambda_eff)/dLambda = {E_gap/lam_eff:.4f}  (halving Lambda2 -> "
          f"lambda_eff {np.sqrt(E_gap*(2*(lam1+lam2/2)-E_gap)):.3f})")


if __name__ == "__main__":
    main()
