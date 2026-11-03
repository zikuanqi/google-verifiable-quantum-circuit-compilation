# -*- coding: utf-8 -*-
"""Benchmark III quantization: continuous (U, w, W) -> angle words -> submission npz.

Stage 1: encode all rows on candidate Givens trees (reverse atan2), decode, and
rank trees by a fast error_2 proxy  ||sum X (x) X - T_shift||_F  via one BLAS
matmul (identical formula, same float32-origin T values).
Stage 2: the top trees get the verifier's own evaluate_dfthc_angle_words
(error_1, error_2, Lambda, E_SOS, E_gap, lambda_eff) and the best accepted
candidate is written in full submission format.

Usage:
    python src/b3_quantize.py results/b3_work_16_24_24.npz --brot 12 13 14 --trees 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
RESULTS = ROOT / "results"
sys.path.insert(0, str(VERIFIERS))
sys.path.insert(0, str(ROOT / "src"))

import b3_cost as bc  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as dm
from dfthcsos.scoreinfra import reference as ref

N = 76
E_UPPER = ref.UPPER_ENERGY


def make_trees(kind: str, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    trees = []
    if kind in ("all", "baseline"):
        base = np.load(ROOT / "data" / "sample_submission" / "dfthcsos" / "submission.npz",
                       allow_pickle=False)
        trees.append(np.array(base["tree_edges"]))
    if kind in ("all", "path"):
        trees.append(np.column_stack([np.arange(N - 1), np.arange(1, N)]))
    if kind in ("all", "balanced"):
        trees.append(np.column_stack([(np.arange(1, N) - 1) // 2, np.arange(1, N)]))
    if kind in ("all", "star"):
        trees.append(np.column_stack([np.zeros(N - 1, dtype=int), np.arange(1, N)]))
    if kind in ("all", "random"):
        for _ in range(count):
            visited = [0]
            edges = []
            for node in range(1, N):
                parent = int(rng.choice(visited))
                edges.append((parent, node))
                visited.append(node)
            trees.append(np.array(edges, dtype=np.int64))
    return trees


def encode_words(vectors: np.ndarray, tree: np.ndarray, brot: int) -> np.ndarray:
    """Reverse atan2 encoding of unit vectors on the Givens tree."""
    A, _ = vectors.shape
    v = np.array(vectors, dtype=np.float64)
    words = np.zeros((A, N - 1), dtype=np.int64)
    scale = 2.0 ** (brot - 1)
    for t in range(N - 2, -1, -1):
        p, c = int(tree[t, 0]), int(tree[t, 1])
        vp, vc = v[:, p], v[:, c]
        theta = np.arctan2(vc, vp)
        words[:, t] = np.rint(theta / np.pi * scale).astype(np.int64) % (1 << brot)
        v[:, p] = np.sqrt(vp * vp + vc * vc)
        v[:, c] = 0.0
    return words


def decoded_X(prob_data: dict, U: np.ndarray, w: np.ndarray, tree: np.ndarray,
              brot: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize two-body rows -> decode -> X_q -> h1'(X_q) eigenvectors -> one-body rows.

    The verifier builds h1_prime from the *decoded* X, so the one-body rows must
    encode h1'(X_q)'s exact eigenvectors (matching its internal consistency).
    """
    ref_data, S, beta_1 = prob_data["ref"], prob_data["S"], prob_data["beta_1"]
    W = prob_data["W"]
    U = U / np.linalg.norm(U, axis=2, keepdims=True)
    words2 = encode_words(U.reshape(-1, N), tree, brot)
    u2 = dm.reconstruct_u_vectors(words2 / (1 << (brot - 1)), tree)
    Uq = u2.reshape(U.shape)
    Xq = np.einsum("tjp,tjq,tjl->tlpq", Uq, Uq, w, optimize=True)
    Xq = np.tril(Xq) + np.swapaxes(np.tril(Xq, k=-1), -1, -2)
    h1q = dm.build_h1_prime(ref_data["h1_thc"], S, beta_1, ref_data["electrons"], W, Xq)
    _, evecs = np.linalg.eigh(h1q)
    words1 = encode_words(evecs.T, tree, brot)
    words = np.vstack([words1, words2])
    return Xq, words


def fast_e2(prob_data: dict, Xq: np.ndarray) -> float:
    XR = Xq.reshape(Xq.shape[0] * Xq.shape[1], N * N)
    G = XR.T @ XR
    D = G - prob_data["T"]
    return float(np.sqrt((D * D).sum()))


def build_submission(U: np.ndarray, w: np.ndarray, W: np.ndarray, tree: np.ndarray,
                     words: np.ndarray, name: Path, shifts: dict,
                     brot: int, batch_size: int) -> None:
    H2_I = W + w.sum(axis=1)
    np.savez(
        name,
        U=U.astype(np.float32),
        W=w.astype(np.float32),
        H2_I=H2_I.astype(np.float32),
        H1_I=shifts["H1_I"].astype(np.float32).reshape(()),
        Hsym_lower=shifts["Hsym_lower"].astype(np.float32),
        tree_edges=tree.astype(np.int64),
        angle_words=words.astype(np.int64),
        brot=np.array([brot], dtype=np.int64),
        batch_size=np.array([batch_size], dtype=np.int64),
    )


def load_state(path: Path):
    """Accept b3_work (U, w, W) or trainer (U, W, H2_I, ...) npz formats."""
    data = np.load(path, allow_pickle=False)
    if "w" in data.files:
        return np.array(data["U"], dtype=np.float64), np.array(data["w"], dtype=np.float64), \
            np.array(data["W"], dtype=np.float64)
    U = np.array(data["U"], dtype=np.float64)
    w = np.array(data["W"], dtype=np.float64)
    W = np.array(data["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    return U, w, W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="results/b3_work_*.npz with U, w, W")
    ap.add_argument("--brot", type=int, nargs="+", default=[12])
    ap.add_argument("--trees", type=int, default=20, help="random trees to try")
    ap.add_argument("--tree-kind", default="all")
    ap.add_argument("--top", type=int, default=3, help="trees to fully verify per brot")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.is_file():
        state_path = RESULTS / args.state
    U, w, W = load_state(state_path)
    R, B, C = w.shape
    print(f"state {args.state}: (R,B,C)=({R},{B},{C})")

    ref_data = ref.load_reference(ref.check_fcidump())
    shifts = ref.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(shifts["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    tei = np.asarray(ref_data["tei"], dtype=np.float64)
    T = np.ascontiguousarray(tei.reshape(N * N, N * N))
    eye = np.eye(N)
    T = T + 0.5 * (S[:, :, None, None] * eye[None, None, :, :]).reshape(N * N, N * N)
    T = T + 0.5 * (eye[:, :, None, None] * S[None, None, :, :]).reshape(N * N, N * N)
    prob_data = dict(ref=ref_data, S=S, beta_1=beta_1, W=W, T=T)

    rng = np.random.default_rng(0)
    trees = make_trees(args.tree_kind, args.trees, rng)
    print(f"trees: {len(trees)}")

    acc = {}
    for brot in args.brot:
        print(f"\n=== brot = {brot} ===")
        t0 = time.time()
        scored = []
        for ti, tree in enumerate(trees):
            Xq, words = decoded_X(prob_data, U, w, tree, brot)
            e2 = fast_e2(prob_data, Xq)
            scored.append((e2, ti, tree, words))
        scored.sort(key=lambda z: z[0])
        print(f"  proxy scan done in {time.time() - t0:.1f}s; "
              f"top e2: {[f'{z[0]:.5f}' for z in scored[:5]]}")
        for rank, (e2, ti, tree, words) in enumerate(scored[: args.top]):
            m = dm.evaluate_dfthc_angle_words(
                words, w, W, S, beta_1, ref_data, brot=brot, tree_edges=tree,
                energy_upper=E_UPPER)
            r = bc.resource(R, B, C, brot)
            ok = (m["error_1"] <= 0.020 and m["error_2"] <= 0.020 and m["gap_in_domain"]
                  and m["Lambda"] > 49.0 and m["E_gap"] > 3.914898
                  and r["angles_within_cap"])
            score = m["lambda_eff"] * r["t_total"] * r["q_total"] if ok else float("inf")
            print(f"  tree {ti:3d} (rank {rank}): e1={m['error_1']:.6f} e2={m['error_2']:.6f} "
                  f"Lam={m['Lambda']:.4f} Eg={m['E_gap']:.4f} leff={m['lambda_eff']:.4f} "
                  f"T={r['t_total']} Q={r['q_total']} score={score:.4e} "
                  f"{'OK' if ok else 'REJECT'}")
            if ok and (brot not in acc or score < acc[brot]["score"]):
                acc[brot] = dict(m=m, words=words, tree=tree, r=r, score=score)

    if not acc:
        print("\nno accepted candidate")
        return
    bb = min(acc, key=lambda b: acc[b]["score"])
    v = acc[bb]
    prefix = args.out_prefix or f"b3_cand_{R}_{B}_{C}_b{bb}"
    out = RESULTS / f"{prefix}.npz"
    build_submission(U, w, W, v["tree"], v["words"], out, shifts, bb, v["r"]["J"])
    print(f"\nwrote {out}  (brot={bb})")
    m = v["m"]
    print(f"final: e1={m['error_1']:.6f} e2={m['error_2']:.6f} Lam={m['Lambda']:.4f} "
          f"Eg={m['E_gap']:.4f} leff={m['lambda_eff']:.4f} "
          f"T={v['r']['t_total']} Q={v['r']['q_total']} score={v['score']:.4e}")


if __name__ == "__main__":
    main()
