#!/usr/bin/env python
"""Fit B1 v3 variants: tolerate domain-check failure, continue."""
import sys
import time

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import numpy as np

import b1v3_model as m


def run(f2, f1, fyr, gy, drop, search=4, k2span=2):
    tag = f"{f2}_{f1}_{fyr}_{gy}_d{'_'.join(map(str, drop))}"
    cfg = m.V3Cfg(f2=f2, f1=f1, fyr=fyr, gy=gy, drop2=drop)
    t0 = time.time()
    tabs = {}
    worst = 0.0
    for e in (0, 1):
        for a in range(128):
            v, k2, k1, k0 = m.fit_segment(cfg, e, a, search=search, k2span=k2span)
            tabs[(e, a)] = (k2, k1, k0)
            worst = max(worst, v)
    print(f"[{tag}] fit {time.time()-t0:.0f}s, metric {worst/2**cfg.fy/2**-24:.4f} x2^-24")
    try:
        lay = m.make_layout(cfg, tabs)
        res = m.full_domain_check(cfg, tabs)
        status = "PASS"
        line = f"worst {res['worst']:.4f} at x={res['worst_x']}, margin {res['margin_units']}"
    except ValueError as ex:
        status = "FAIL"
        lay = None
        line = str(ex)[:120]
    print(f"[{tag}] domain {status}: {line}")
    if status == "PASS":
        out = f"/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/results/b1v3_tabs_{tag}.npz"
        np.savez(out,
                 table=np.array([tabs[(e, a)] for e in (0, 1) for a in range(128)], dtype=np.int64),
                 config=np.array([f2, f1, fyr, gy]), drop=np.array(list(drop)))
        print(f"[{tag}] saved {out}")
    return status


if __name__ == "__main__":
    run(8, 18, 27, 1, (5,))
    run(7, 19, 27, 1, (5,))
    run(8, 19, 27, 1, (5, 6), search=8, k2span=3)
    run(8, 18, 27, 1, (5, 6), search=8, k2span=3)
