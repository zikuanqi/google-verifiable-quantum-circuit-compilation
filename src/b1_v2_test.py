"""Quick B1 v2 circuit test: simulate a spread of x values, compare phase with
the v2 fixed-point model, and check scratch cleanliness. Runs in WSL."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import math
import numpy as np
import kickmix as km

import b1_v2_model as v2m
import b1_v2_build as b1b

N_P = 10
MAX_X = 3 * (2**N_P - 2)**2
WIDTH = 2 * N_P + 2


def main():
    prec = v2m.V2Precision(K=32, degree=2, coef_bits=17, delta_bits=12,
                           square_bits=28, product_bits=28, result_bits=28)
    circuit = b1b.build_v2_circuit(N_P, prec)
    print(f"qubits={circuit.num_qubits} toffoli={circuit.operation_statistics().num_toffoli_like}")

    # spread of x values covering all s and intervals
    xs = []
    for s in range(0, 11):
        for off in (0, 1, 5, 100, 2**18):
            x = (2**20 + off) >> (2 * s)
            if x >= 1:
                xs.append(x)
    xs = sorted(set(xs))
    xs = np.array(xs, dtype=np.uint64)

    y1_model = v2m.v2_model(N_P, prec, xs.astype(np.int64))
    F = prec.output_bits(N_P)

    sim = km.Simulator(batch_size=max(1, len(xs)), debug_print=False,
                       counting_operations=True, seed=12345)
    sim.clear_for_shot()
    x_reg = km.array([km.q(i) for i in range(WIDTH)])
    sim.write_across_shots(x_reg, ((xs[:, None] >> np.arange(WIDTH, dtype=np.uint64)[None, :]) & 1 == 1).T)
    sim.do(circuit)
    got = sim.read_across_shots(x_reg)  # (num_qubits, num_shots)
    got_x = (got.T.astype(np.uint64) * (1 << np.arange(WIDTH, dtype=np.uint64))[None, :]).sum(axis=1)
    bad = np.flatnonzero(got_x != xs)
    if len(bad):
        print("X NOT RESTORED at", xs[bad][:5])
        sys.exit(1)
    scratch = km.array([km.q(i) for i in range(WIDTH, circuit.num_qubits)])
    if len(scratch):
        dirty = sim.read_across_shots(scratch).any(axis=1)  # per-scratch-qubit
        if dirty.any():
            q = scratch[int(np.argmax(dirty))]
            print("DIRTY SCRATCH qubit", q)
            qbits = sim.read_across_shots(scratch)
            bad_shots = np.flatnonzero(qbits.any(axis=0))
            print("dirty shots (indices):", bad_shots[:5], "x=", xs[bad_shots[:5]])
            sys.exit(1)
    for i in range(len(xs)):
        ph = sim.read_shot_phase(i) / 2.0  # read_shot_phase returns half-turns
        target = (y1_model[i] / 2**F) % 1.0
        err = min(abs(ph - target), 1 - abs(ph - target))
        tol = 2**-24 / math.sqrt(float(xs[i]))
        print(f"x={xs[i]}: ph={ph:.9f} target={target:.9f} err={err:.3e} tol={tol:.3e}")
        if err > tol:
            print(f"PHASE FAIL x={xs[i]}: err={err:.3e} tol={tol:.3e}")
            sys.exit(1)
    print(f"all {len(xs)} x values OK")


if __name__ == "__main__":
    main()
