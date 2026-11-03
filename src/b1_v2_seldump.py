"""Dump the _select_v2 forward+reverse circuit as .kmx text for inspection."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import kickmix as km

import b1_v2_build as b1b

WIDTH = 22
K = 24
B = 5


def main():
    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(WIDTH, name='x')
    onehot = builder.alloc_qubits(K)
    anc = builder.alloc_qubits(B - 1)
    b1b._select_v2(builder, False, x, onehot, anc)
    b1b._select_v2(builder, True, x, onehot, anc)
    for reg in (onehot, anc):
        builder.free(reg)
    circuit = builder.finish_circuit()
    text = str(circuit)
    lines = text.splitlines()
    print(f"total lines: {len(lines)}")
    # print the first 120 non-header lines
    for line in lines[:130]:
        print(line)


if __name__ == "__main__":
    main()
