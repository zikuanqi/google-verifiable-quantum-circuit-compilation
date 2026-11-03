"""Verify the reverse-BM identities the builder relies on:

(a) scaled identity on rep rounds: sigma0_post == inv(delta)  (sigma0 * delta == 1)
(b) clear-conv identity on rep rounds: conv(shift(tau_pre), S_r) == delta
(c) sigma0_post == sigma0_pre on non-rep rounds (retain/no-op)
(d) rep rounds have d7_pre == 0 (no push on rep rounds)
"""
import sys
import numpy as np

sys.path.insert(0, "src")
import b2_model as M

gf = M.GF12()
dist = M.DQIDistribution(m=4095, n=70, degree=12, r=2016, gf=gf)
rng = np.random.default_rng(7)
cutoffs = [min(r, 35) for r in range(1, 71)]

ns = 2000
fail_a = fail_b = fail_c = fail_d = 0
n_rep = 0
bad_a = []
for s in range(ns):
    rec = dist.sample(rng)
    syn = rec["syndromes"]
    st = M.BMState(gf, cutoffs, n=70, caps=3)
    for r in range(1, 71):
        sigma_pre = list(st.sigma)
        tau_pre = list(st.tau)
        d7_pre = 1 if st.d < 0 else 0
        delta, rep = st.round_fwd(r, syn)
        if rep:
            n_rep += 1
            # (a)
            if gf.mul(st.sigma[0], delta) != 1:
                fail_a += 1
                if len(bad_a) < 5:
                    bad_a.append((s, r))
            # (b): shift(tau_pre) is tau_pre cyclically left-rotated by one word
            shifted = [tau_pre[-1]] + tau_pre[:-1]
            acc = 0
            for i in range(min(r, 36)):
                acc ^= gf.mul(shifted[i], syn[r - 1 - i])
            # note: builder's conv range is i in 0..K, K=min(K_r, r-1); K_r=cutoffs[r-1];
            # shifted[i] for i > Km-1 should be 0 on correct paths anyway.
            if acc != delta:
                fail_b += 1
            # (d)
            if d7_pre != 0:
                fail_d += 1
        else:
            # (c)
            if st.sigma[0] != sigma_pre[0]:
                fail_c += 1

print(f"samples={ns} rep_rounds={n_rep}")
print(f"(a) sigma0*delta==1 on rep: fail={fail_a} {bad_a[:3]}")
print(f"(b) conv(shift(tau_pre),S)==delta on rep: fail={fail_b}")
print(f"(c) sigma0 unchanged on non-rep: fail={fail_c}")
print(f"(d) rep rounds always d>=0 pre-update: fail={fail_d}")
