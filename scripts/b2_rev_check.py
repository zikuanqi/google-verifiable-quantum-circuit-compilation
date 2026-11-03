#!/usr/bin/env python
"""Quick check: the tau-based detection's failure modes vs the stack-tag rule (WSL)."""
import sys

sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/src")

import b2_model as m

gf = m.GF12()
n = 70
dist = m.DQIDistribution(m=4095, n=n, degree=m.DEGREE, r=2016, gf=gf)
import numpy as np

rng = np.random.default_rng(7)
cutoffs = [min(r, 35) for r in range(1, n + 1)]

# instrument: for each forward round record (delta, rep, d_old, tau_pre_35, tau_pre_34,
# sigma_pre_0, sigma_pre_35); then verify: tau-based rule would miss exactly when
# sigma_pre_0 == sigma_pre_35 on a rep round (post-swap tau0 == tau35).
misses_tau_rule = 0
false_pos_tau_rule = 0
rounds = 0
stack_rule_ok = 0
for it in range(4000):
    s = dist.sample(rng)
    bm = m.BMState(gf, cutoffs, n, caps=3)
    # forward with per-round snapshots
    snap = []
    for r in range(1, n + 1):
        pre = dict(s0=bm.sigma[0], s35=bm.sigma[35], t0=bm.tau[0], t35=bm.tau[35], d=bm.d)
        delta, rep = bm.round_fwd(r, s['syndromes'])
        snap.append((pre, delta, rep))
    # reverse with the tau-based rule (replayed on the model state)
    bm2 = m.BMState(gf, cutoffs, n, caps=3)
    for r in range(1, n + 1):
        bm2.round_fwd(r, s['syndromes'])
    for r in range(n, 0, -1):
        pre, delta, rep_true = snap[r - 1]
        d_rev = bm2.d - 1
        # tau-based rule: rep = (tau0 != 0) and (d_rev <= 0) and (tau0 != tau35)
        t0, t35 = bm2.tau[0], bm2.tau[35]
        rep_tau = (t0 != 0) and (d_rev <= 0) and (t0 != t35)
        # stack-tag rule
        pushed = bool(bm2.stack) and bm2.stack[-1][0] == r
        rep_tag = (d_rev < 0) and not pushed
        rounds += 1
        if rep_tau != rep_true:
            if rep_true:
                misses_tau_rule += 1
                # on a rep round the post-forward tau = pre-sigma: coincidence = s0==s35
                assert pre['s0'] == pre['s35'], (pre['s0'], pre['s35'])
            else:
                false_pos_tau_rule += 1
                # false positive: pre-tau35 == pre-tau34 != 0 (shifted wrap)
                assert pre['t35'] == pre['t34'] and pre['t35'] != 0, (pre['t35'], pre['t34'])
        if rep_tag != rep_true:
            print("STACK RULE MISMATCH", r, rep_tag, rep_true, "d_rev", d_rev, "pushed", pushed)
        else:
            stack_rule_ok += 1
        bm2.round_rev(r, s['syndromes'])
print(f"rounds={rounds}")
print(f"tau-rule misses (replace branch漏检): {misses_tau_rule}  (~{misses_tau_rule/rounds:.5f} per round)")
print(f"tau-rule false positives: {false_pos_tau_rule}")
print(f"stack-tag rule mismatches: {rounds - stack_rule_ok}")
