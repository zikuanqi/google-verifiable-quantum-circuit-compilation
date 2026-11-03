#!/usr/bin/env python
"""Fit B1 v3 signed-digit coefficients and check the full domain (WSL)."""
import sys
import time

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import numpy as np

import b1v3_model as m


def main():
    f2, f1, fyr, gy = 8, 19, 27, 1
    drop = (5,)
    if len(sys.argv) > 1:
        parts = sys.argv[1].split("_")
        f2, f1, fyr, gy = map(int, parts[:4])
        drop = tuple(int(t) for t in parts[4].split("d")[1].split(","))
    tag = f"{f2}_{f1}_{fyr}_{gy}_d{'_'.join(map(str, drop))}"
    cfg = m.V3Cfg(f2=f2, f1=f1, fyr=fyr, gy=gy, drop2=drop)
    t0 = time.time()
    tabs, worst = m.fit_all(cfg, verbose=False)
    print(f"fit done {time.time()-t0:.1f}s, worst metric {worst/2**cfg.fy:.4e} "
          f"({worst/2**cfg.fy/2**-24:.4f} x 2^-24)")
    lay = m.make_layout(cfg, tabs)
    res = m.full_domain_check(cfg, tabs)
    print(f"worst error/tol = {res['worst']:.4f} at x={res['worst_x']}")
    print(f"margin units = {res['margin_units']}")
    out = f"/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1v3_tabs_{tag}.npz"
    np.savez(out,
             table=np.array([tabs[(e, a)] for e in (0, 1) for a in range(128)], dtype=np.int64),
             config=np.array([f2, f1, fyr, gy]), drop=np.array(list(drop)),
             worst=res["worst"], worst_x=res["worst_x"])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
