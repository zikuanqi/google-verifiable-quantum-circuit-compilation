# -*- coding: utf-8 -*-
"""B3: test the selective-copy hypothesis for (65, 60, C<60).

For the baseline factorization, measure each (rho, c) copy's contribution to
fitting the two-body tensor, keep the top-C copies (zeroing the rest), and
evaluate error_2 / Lambda / E_gap / lambda_eff with the official math.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "verifiers"))
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402
from dfthcsos.scoreinfra import reference as ref  # noqa: E402


def main():
    ref_data = ref.load_reference(ref.check_fcidump())
    fact = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                   allow_pickle=False)
    params = dict(fact)
    params.update(ref.load_fixed_shifts())
    w = np.asarray(params["W"], dtype=np.float64)   # (65, 60, 60)
    W = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)  # identity offsets
    S = np.asarray(dm.symmetric_from_lower(params["Hsym_lower"], 76), dtype=np.float64)
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(S))
    U = np.array(fact["U"], dtype=np.float64)

    # per-copy contribution: ||X^{rc}||_F^2 = ||sum_b w_b u_b u_b^T||_F^2
    X = dm.reconstruct_x(U, w)  # (65, 60, 76, 76)
    contrib = np.sum(X**2, axis=(2, 3))  # (65, 60)
    flat = contrib.ravel()
    order = np.argsort(-flat)  # copy (r,c) pairs by contribution
    print("top-10 contribution share:", flat[order[:10]].sum() / flat.sum())
    print("top-30 contribution share:", flat[order[:30]].sum() / flat.sum())
    print("top-60 contribution share:", flat[order[:60]].sum() / flat.sum())

    for C in (30, 40, 50):
        # keep the top-C copies globally: mask w and W accordingly
        keep = np.zeros(len(flat), dtype=bool)
        keep[order[:C]] = True
        keep = keep.reshape(65, 60)
        w2 = w * keep[:, :, None]
        W2 = W * keep
        X2 = dm.reconstruct_x(U, w2)
        X2 = np.tril(X2) + np.swapaxes(np.tril(X2, k=-1), -1, -2)
        h2 = dm.reconstruct_h2(X2) - ref_data["tei"]
        diag = np.arange(76)
        h2[:, :, diag, diag] -= 0.5 * S[:, :, None]
        h2[diag, diag, :, :] -= 0.5 * S[None, :, :]
        e2 = float(np.linalg.norm(h2.ravel()))
        # metrics with the kept copies (W2 shape stays (65,60) => C=60 for the
        # metric formulas; zero copies contribute 0)
        h1_thc = ref_data["h1_thc"]
        h1p = dm.build_h1_prime(h1_thc, S, beta_1, ref_data["electrons"], W2, X2)
        evals = np.linalg.eigvalsh(h1p)
        lam1 = float(np.sum(np.abs(evals)))
        tr = float(np.sum(evals))
        lam2 = dm.lambda_h2(w2, W2)
        Lambda = lam1 + lam2
        E_SOS = (ref_data["ecore"] - ref_data["electrons"] * beta_1
                 - (lam1 - tr) - 0.5 * np.sum(W2**2))
        E_gap = ref.UPPER_ENERGY - E_SOS
        in_dom = 0.0 < E_gap < 2 * Lambda
        leff = float(np.sqrt(E_gap * (2 * Lambda - E_gap))) if in_dom else float("nan")
        print(f"C_kept={C}: e2={e2:.5f} Lambda={Lambda:.2f} E_gap={E_gap:.4f} "
              f"lambda_eff={leff:.3f} e1=?")


if __name__ == "__main__":
    main()
