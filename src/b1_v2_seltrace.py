"""Step-by-step trace of _select_v2 for x=2^20 (selector 8, interval 0)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import numpy as np
import kickmix as km

import b1_v2_build as b1b

WIDTH = 22
K = 24
B = 5


def build(fwd_only):
    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(WIDTH, name='x')
    onehot = builder.alloc_qubits(K)
    anc = builder.alloc_qubits(B - 1)
    b1b._select_v2(builder, False, x, onehot, anc)
    if not fwd_only:
        b1b._select_v2(builder, True, x, onehot, anc)
    for reg in (onehot, anc):
        builder.free(reg)
    return builder.finish_circuit()


def run(circuit, xv):
    xs = np.array([xv], dtype=np.uint64)
    sim = km.Simulator(batch_size=1, debug_print=False, counting_operations=True, seed=7)
    sim.clear_for_shot()
    x_reg = km.array([km.q(i) for i in range(WIDTH)])
    sim.write_across_shots(x_reg, ((xs[:, None] >> np.arange(WIDTH, dtype=np.uint64)[None, :]) & 1 == 1).T)
    sim.do(circuit)
    got = sim.read_across_shots(km.array([km.q(i) for i in range(WIDTH, WIDTH + K + B - 1)]))
    return got[:, 0]


def main():
    xv = 2**20
    fwd = run(build(fwd_only=True), xv)
    full = run(build(fwd_only=False), xv)
    onehot_f = fwd[:K]
    anc_f = fwd[K:]
    onehot_r = full[:K]
    anc_r = full[K:]
    print("forward onehot set at:", np.flatnonzero(onehot_f).tolist())
    print("forward anc:", anc_f.tolist())
    print("reverse onehot still set:", np.flatnonzero(onehot_r).tolist())
    print("reverse anc:", anc_r.tolist())
    print("phase:", run(build(fwd_only=False), xv))


if __name__ == "__main__":
    main()
