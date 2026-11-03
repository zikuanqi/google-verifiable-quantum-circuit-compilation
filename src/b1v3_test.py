"""Quick B1 v3 circuit test: simulate a spread of x values, compare the phase with
the exact v3 model, and check scratch cleanliness. Runs in WSL."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import math

import numpy as np
import kickmix as km

import b1v3_build as b3b
import b1v3_model as v3m

N_P = 10
MAX_X = 3 * (2**N_P - 2)**2
W = 22


def main():
    npz_path = (sys.argv[1] if len(sys.argv) > 1 else
                "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1v3_tabs_8_19_27_1_d5.npz")
    kmb_path = (sys.argv[2] if len(sys.argv) > 2 else
                "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1v3_test.kmb")
    npz = np.load(npz_path)
    tab = npz["table"]
    tabs = {(e, a): tuple(int(v) for v in tab[a + 128 * e]) for e in (0, 1) for a in range(128)}
    cfg = v3m.V3Cfg(f2=8, f1=19, fyr=27, gy=1, drop2=(5,))
    lay = v3m.make_layout(cfg, tabs, verbose=True)
    circuit = km.Circuit.from_file(kmb_path)
    st = circuit.operation_statistics()
    print(f"qubits={circuit.num_qubits} toffoli={st.num_toffoli_like} t={st.num_t_like} "
          f"rot={st.num_z_pow(min_clifford_hierarchy_level=4)}")

    # spread of x covering all shifts and segments + boundary values
    xs = set()
    for s in range(0, 22):
        for off in (0, 1, 5, 100, 2**18):
            x = (2**20 + off) >> s
            if x >= 1:
                xs.add(int(x))
    xs |= {1, 2, 3, MAX_X, MAX_X - 1, 1034666, 2**20, 2**21 - 1, 2**21, 3 * 2**20}
    xs = np.array(sorted(xs), dtype=np.uint64)

    # model phases for these x
    s = 21 - np.floor(np.log2(xs.astype(np.float64))).astype(np.int64)
    xn = xs.astype(np.int64) << s
    e = s & 1
    a = (xn >> 14) & 127
    d = xn & 16383
    k2, k1, k0 = tab[128 * e + a].T
    sig = 2 * ((d[:, None] >> np.arange(14)) & 1) - 1
    r = np.full_like(d, k1, dtype=np.int64)
    for i in cfg.rrows():
        r = r - sig[:, i] * v3m.rd(k2, cfg.sh2(i))
    q = np.zeros_like(d, dtype=np.int64)
    for i in reversed(range(14)):
        q = q + sig[:, i] * v3m.rd(r, cfg.sh1(i))
    y = (k0 - q) >> cfg.gy
    units = (y.astype(np.uint64) << np.uint64(27 + s // 2)) & np.uint64((1 << 64) - 1)

    sim = km.Simulator(batch_size=max(1, len(xs)), debug_print=False,
                       counting_operations=True, seed=12345)
    sim.clear_for_shot()
    x_reg = km.array([km.q(i) for i in range(W)])
    bits = ((xs[:, None] >> np.arange(W, dtype=np.uint64)[None, :]) & 1 == 1).T
    sim.write_across_shots(x_reg, bits)
    sim.do(circuit)

    got = sim.read_across_shots(x_reg)
    got_x = (got.T.astype(np.uint64) * (1 << np.arange(W, dtype=np.uint64))[None, :]).sum(axis=1)
    bad = np.flatnonzero(got_x != xs)
    if len(bad):
        print("X NOT RESTORED at", xs[bad][:5])
        sys.exit(1)

    scratch = km.array([km.q(i) for i in range(W, circuit.num_qubits)])
    if len(scratch):
        dirty = sim.read_across_shots(scratch).any(axis=1)
        if dirty.any():
            q = scratch[int(np.argmax(dirty))]
            print("DIRTY SCRATCH qubit", q)
            qbits = sim.read_across_shots(scratch)
            bad_shots = np.flatnonzero(qbits.any(axis=0))
            print("dirty shots:", bad_shots[:5], "x=", xs[bad_shots[:5]])
            sys.exit(1)

    worst = 0.0
    for i in range(len(xs)):
        ph = sim.read_shot_phase(i) / 2.0          # turns
        target_turns = (units[i] / 2**64) % 1.0
        err = min(abs(ph - target_turns), 1 - abs(ph - target_turns))
        tol = 2**-24 / math.sqrt(float(xs[i]))
        worst = max(worst, err / tol)
        if err > tol:
            print(f"PHASE FAIL x={xs[i]}: err={err:.3e} tol={tol:.3e}")
            sys.exit(1)
    print(f"all {len(xs)} x values OK, worst ratio {worst:.4f}")


if __name__ == "__main__":
    main()
