"""Build the ANGLE_k circuit family for a dfthcsos submission npz.

Used inside the Kaggle submission kernel: takes the factorization npz (with
angle_words, tree_edges, and optional brot / batch_size attributes stored as
plain arrays) and writes angle_k_*.kmb files.

This adapts the official example builder
(data/verifiers/dfthcsos/example_submission/angle_k_reference_circuit.py):
an ordinary QROM per batch with a segment-tree unary iteration over the
`selection` register. That file imports dfthcsos.scoreinfra (kickmix present);
in the kernel we only need kickmix, so the builder is reimplemented here.
"""
from __future__ import annotations

import math
import os

import kickmix as km
import numpy as np


def batch_deltas(angle_words, batch_size):
    words = np.asarray(angle_words)
    num_vectors, num_angles = words.shape
    num_batches = max(1, math.ceil(num_angles / batch_size))
    padded = np.zeros((num_vectors, num_batches * batch_size), dtype=np.int64)
    padded[:, :num_angles] = words
    batches = padded.reshape(num_vectors, num_batches, batch_size).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas


def build_angle_k_circuit(delta, brot):
    delta = np.asarray(delta, dtype=np.int64)
    num_rows, batch_size = delta.shape
    num_bits = (num_rows - 1).bit_length()
    builder = km.CircuitBuilder()
    selection = builder.create_quantum_register(num_bits, "selection")
    words = [builder.create_quantum_register(brot, f"word_{j}") for j in range(batch_size)]
    ancillas = builder.alloc_qubits(num_bits - 1) if num_bits > 1 else []
    bits = [selection[i] for i in reversed(range(num_bits))]

    def load(row, control):
        for j in range(batch_size):
            for bit in reversed(range(brot)):
                if (int(delta[row, j]) >> bit) & 1:
                    builder.cx(control, words[j][bit])

    def node(control, level, lo, hi):
        if lo >= num_rows:
            return
        if hi <= num_rows and (hi - lo == 1 or np.all(delta[lo:hi] == delta[lo, 0])):
            load(lo, control)
            return
        mid = (lo + hi) // 2
        if num_rows <= mid:
            node(control, level + 1, lo, mid)
            return
        bit, ancilla = bits[level], ancillas[level - 1]
        builder.x(bit)
        builder.init_and(control, bit, ancilla)
        builder.x(bit)
        node(ancilla, level + 1, lo, mid)
        builder.cx(control, ancilla)
        node(ancilla, level + 1, mid, hi)
        builder.del_and(control, bit, ancilla)

    half = 1 << (num_bits - 1)
    builder.x(bits[0])
    node(bits[0], 1, 0, half)
    builder.x(bits[0])
    node(bits[0], 1, half, 2 * half)
    return builder.finish_circuit()


def get_attr(fact, key, default=None):
    """Read an integer attribute stored as a length-1 array in the npz."""
    if key in fact.files:
        arr = np.asarray(fact[key])
        return int(arr.reshape(-1)[0])
    return default


def build_family(fact, outdir):
    """Write angle_k_0.kmb ... for the npz's angle_words table."""
    words = np.asarray(fact["angle_words"])
    brot = get_attr(fact, "brot", 10)
    batch_size = get_attr(fact, "batch_size", 5)
    print(f"angle word table {words.shape}, brot={brot}, J={batch_size}")
    deltas = batch_deltas(words, batch_size)
    paths = []
    for k, delta in enumerate(deltas):
        circuit = build_angle_k_circuit(delta, brot)
        path = os.path.join(outdir, f"angle_k_{k}.kmb")
        circuit.to_file(path, format="kmb")
        st = circuit.operation_statistics()
        print(f"  {os.path.basename(path)}: {st.num_toffoli_like} Toffolis, "
              f"{circuit.num_qubits} qubits")
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    fact = np.load(sys.argv[1], allow_pickle=False)
    build_family(fact, sys.argv[2] if len(sys.argv) > 2 else ".")
