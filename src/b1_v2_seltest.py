"""Isolate the _select_v2 stage: forward+reverse, check onehot cleanup."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import numpy as np
import kickmix as km

import b1_v2_build as b1b

N_P = 10
WIDTH = 22


def main():
    K = 24
    B = 5
    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(WIDTH, name='x')
    onehot = builder.alloc_qubits(K)
    anc = builder.alloc_qubits(B - 1)
    b1b._select_v2(builder, False, x, onehot, anc)
    b1b._select_v2(builder, True, x, onehot, anc)
    for reg in (onehot, anc):
        builder.free(reg)
    circuit = builder.finish_circuit()

    xs = np.array([1, 4, 16, 2**20, 2**21, 3 * (2**10 - 2)**2, 1294339], dtype=np.uint64)
    sim = km.Simulator(batch_size=len(xs), debug_print=False, counting_operations=True, seed=1)
    sim.clear_for_shot()
    x_reg = km.array([km.q(i) for i in range(WIDTH)])
    sim.write_across_shots(x_reg, ((xs[:, None] >> np.arange(WIDTH, dtype=np.uint64)[None, :]) & 1 == 1).T)
    sim.do(circuit)
    got = sim.read_across_shots(km.array([km.q(i) for i in range(WIDTH, WIDTH + K + B - 1)]))
    dirty = got.any(axis=0)
    print("x:", xs.tolist())
    print("dirty after full cycle:", dirty.tolist())
    for i, xv in enumerate(xs):
        s = (WIDTH - (int(np.log2(xv)) + 1)) // 2
        x_norm = int(xv) << (2 * s)
        sel = x_norm >> 17
        i_exp = sel - 8
        print(f"x={xv}: s={s} x_norm={x_norm} sel={sel} interval={i_exp}")


if __name__ == "__main__":
    main()
