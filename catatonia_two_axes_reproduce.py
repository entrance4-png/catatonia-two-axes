#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
catatonia_two_axes_reproduce.py  (v3, Journal of Mathematical Biology version)

Self-contained reproduction, figure and self-check script for

    Catatonia recovery separates two intervention axes that constrain
    electroconvulsive therapy
    Hiroki Saito

Every numerical value printed in the journal manuscript is recomputed here and
checked against the value as printed (84 checks).  The script also carries the
secondary analysis of published aggregate data, draws the five figures and
writes their source data.  It has no inputs.

Version history
---------------
v1  preprint version (PsyArXiv d6knq): computations, figures, source data,
    and the preprint text with its journal-compliance audit.
v2  section 13 added (run_review_additions): the initial-condition example of
    Remark 2, the passage-time constant of Proposition 8(ii), and the exact
    two-stage inversion of Proposition 9 with its sensitivity, each self-checked.
v3  the preprint text, its audit and the docx renderer were removed; the
    journal manuscript is maintained separately as
    manuscript/catatonia_two_axes_JMB.docx.  Computations, figures and checks
    are unchanged from v2.

Model (Saito 2026a, Research Square rs-10620384)
------------------------------------------------
    g(r,Q) = r (1 - r) (r - a(Q))
    dr0/dt = g(r0,Q) + u(t)
    drk/dt = g(rk,Q) + kappa_k (r_{k-1} - r_k),      k = 1..3
    dQ/dt  = eps [ chi_+(z) C(r) (1 - Q) - rho chi_-(z) Q ],   z = r0 - a(Q)
    C(r)   = prod_j r_j,   a(Q) = a0 - (a0 - a1) Q^p,  p > 1
    chi_+-(z) = exp(-s / z^2) on its own half-line, 0 elsewhere

Usage
-----
    python3 catatonia_two_axes_reproduce.py                # full run
    python3 catatonia_two_axes_reproduce.py --quick        # reduced sample sizes
    python3 catatonia_two_axes_reproduce.py --no-figures   # values and self-check only
    python3 catatonia_two_axes_reproduce.py --figures-only # redraw from two_axes_results.json
    python3 make_jmb_figures.py                            # journal figure formats

Outputs
-------
    Fig1.png ... Fig5.png                    the five figures (300 dpi; see make_jmb_figures.py)
    two_axes_source_data.xlsx                one sheet per panel (Online Resource 2)
    two_axes_results.json                    every computed quantity

Requires numpy and scipy.  Matplotlib and openpyxl are needed for the figures and
the workbook; both are optional.
"""

from __future__ import annotations
import argparse, json, math, re, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# 0.  Illustrative parameter set of ref. 4  (unfitted)
# ----------------------------------------------------------------------------
A0, A1, P_EXP, EPS, RHO = 0.60, 0.15, 2.0, 0.02, 1.0
KAPPA_REF = 0.6
S_NARROW, S_WIDE = 0.01, 1.0
RTOL, ATOL = 1e-10, 1e-13

RESULTS: dict = {}
CHECKS: list = []


def check(name, got, expected, tol, note=""):
    """Record a comparison between a recomputed value and the value as printed."""
    ok = abs(got - expected) <= tol
    CHECKS.append(dict(name=name, got=float(got), printed=float(expected),
                       tol=float(tol), ok=bool(ok), note=note))
    flag = "OK  " if ok else "FAIL"
    print(f"  [{flag}] {name:<52s} got {got:.10g}  printed {expected:.10g}")
    return ok


def a_of_Q(Q, a0=A0, a1=A1, p=P_EXP):
    return a0 - (a0 - a1) * Q ** p


def g(r, a):
    return r * (1.0 - r) * (r - a)


def chi_plus(z, s):
    return math.exp(-s / z ** 2) if z > 0 else 0.0


def chi_minus(z, s):
    return math.exp(-s / z ** 2) if z < 0 else 0.0


# ----------------------------------------------------------------------------
# 1.  Confinement below threshold  (maximum principle, main text and Fig. 1b)
# ----------------------------------------------------------------------------
def draw_weights(rng, n, family):
    """Mixture placing mass at 0 and at 0.5, 3, 30 times a uniform draw on (0,1]."""
    base = rng.uniform(1e-9, 1.0, size=(n, n))
    scale = rng.choice([0.0, 0.5, 3.0, 30.0], size=(n, n), p=[1/3, 2/9, 2/9, 2/9])
    W = base * scale
    if family == "chain":          # feed-forward: strict sub-diagonal only
        M = np.zeros((n, n));  M[np.arange(1, n), np.arange(0, n - 1)] = 1.0
        W = W * M
    elif family == "reverse":      # purely reverse: strict upper triangle
        W = np.triu(W, 1)
    elif family == "symmetric":
        W = np.tril(W, -1); W = W + W.T
    elif family == "general":
        np.fill_diagonal(W, 0.0)
    return W


def run_confinement(n_per_family=100, seed=20260814):
    rng = np.random.default_rng(seed)
    fams = ["chain", "symmetric", "reverse", "general"]
    reached, max_rise = 0, -np.inf
    for fam in fams:
        for _ in range(n_per_family):
            n = int(rng.integers(3, 7))
            W = draw_weights(rng, n, fam)
            r0 = rng.uniform(0.0, A0 * 0.999, size=n)
            M0 = r0.max()

            def f(t, r):
                r = np.clip(r, 0.0, 1.0)
                return g(r, A0) + (W @ r - W.sum(axis=1) * r)

            sol = solve_ivp(f, [0, 2000.0], r0, method="LSODA",
                            rtol=1e-9, atol=1e-12, dense_output=True)
            traj = sol.sol(np.linspace(0, 2000.0, 4001))
            M = traj.max(axis=0)
            if M.max() >= A0:
                reached += 1
            max_rise = max(max_rise, M.max() - M0)
    return dict(n_systems=4 * n_per_family, reached_threshold=reached,
                largest_rise_of_running_max=float(max_rise))


def run_confinement_heterogeneous(n_graphs=200, seed=7):
    rng = np.random.default_rng(seed)
    reached = 0
    for _ in range(n_graphs):
        n = int(rng.integers(3, 7))
        W = draw_weights(rng, n, "general")
        a = rng.uniform(A1, A0, size=n)
        r0 = rng.uniform(0.0, a.min() * 0.999, size=n)

        def f(t, r):
            r = np.clip(r, 0.0, 1.0)
            return r * (1 - r) * (r - a) + (W @ r - W.sum(axis=1) * r)

        y = solve_ivp(f, [0, 2000.0], r0, method="LSODA",
                      rtol=1e-9, atol=1e-12).y
        if np.any(y.max(axis=1) >= a.min()):
            reached += 1
    return dict(n_graphs=n_graphs, reached_min_threshold=reached)


def run_reverse_counterexample():
    """r = (0.30, 0.90, 0.90, 0.90): the maximum is NOT below a, and reverse
    coupling of 0.10 then converts the root."""
    def f(t, r, kf, kr):
        r = np.clip(r, 0, 1); o = np.zeros(4)
        for k in range(4):
            o[k] = g(r[k], A0)
            if k > 0:
                o[k] += kf * (r[k - 1] - r[k])
            if k < 3:
                o[k] += kr * (r[k + 1] - r[k])
        return o
    y0 = [0.30, 0.90, 0.90, 0.90]
    both = solve_ivp(f, [0, 3e4], y0, args=(0.01, 0.10), method="LSODA",
                     rtol=RTOL, atol=ATOL).y[:, -1]
    rev = solve_ivp(f, [0, 3e4], y0, args=(0.0, 0.10), method="LSODA",
                    rtol=RTOL, atol=ATOL).y[:, -1]
    return dict(forward001_reverse010=np.round(both, 6).tolist(),
                reverse_only=np.round(rev, 6).tolist())


def run_root_kappa_independence():
    """On the forward chain kappa multiplies zero in the root equation."""
    def f(t, r, k):
        r = np.clip(r, 0, 1); o = np.zeros(4); o[0] = g(r[0], A0)
        for j in range(1, 4):
            o[j] = g(r[j], A0) + k * (r[j - 1] - r[j])
        return o
    grid = np.linspace(0, 200, 20001)
    roots = []
    for k in (0.6, 3.0, 30.0):
        s = solve_ivp(f, [0, 200], [0.30, 0.20, 0.10, 0.05], args=(k,),
                      method="LSODA", rtol=RTOL, atol=ATOL, dense_output=True)
        roots.append(s.sol(grid)[0])
    d = max(np.max(np.abs(roots[0] - roots[1])), np.max(np.abs(roots[0] - roots[2])))
    return dict(max_pointwise_root_difference=float(d))


# ----------------------------------------------------------------------------
# 2.  Root pulse: the crossing criterion is r0(tau) > a0   (Fig. 1c)
# ----------------------------------------------------------------------------
def chain_rhs(t, r, kappa, a, u=0.0, tau=0.0):
    r = np.clip(r, 0.0, 1.0)
    o = np.zeros(4)
    o[0] = g(r[0], a) + (u if t <= tau else 0.0)
    for k in range(1, 4):
        o[k] = g(r[k], a) + kappa * (r[k - 1] - r[k])
    return o


def pulse_outcome(A, tau, kappa=KAPPA_REF, a=A0, T=4000.0):
    s = solve_ivp(chain_rhs, [0, T], [0.0, 0.0, 0.0, 0.0],
                  args=(kappa, a, A, tau), method="LSODA",
                  rtol=RTOL, atol=ATOL, dense_output=True)
    return s.sol(tau)[0], s.y[3, -1] > 0.5


def run_pulse_sweep():
    rows = []
    for tau in (0.25, 0.5, 1.0, 2.0, 5.0):
        lo, hi = 1e-4, 50.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if pulse_outcome(mid, tau)[1]:
                hi = mid
            else:
                lo = mid
        Ac = 0.5 * (lo + hi)
        cell = []
        for m in (0.8, 0.99, 1.01, 1.2):
            r0tau, conv = pulse_outcome(m * Ac, tau)
            cell.append(dict(mult=m, r0_at_tau=float(r0tau), converted=bool(conv),
                             agrees=bool(conv == (r0tau > A0))))
        rows.append(dict(tau=tau, critical_amplitude=float(Ac), cells=cell))
    agree = all(c["agrees"] for row in rows for c in row["cells"])
    # boundary in the direct-set sweep
    def outcome_from_root(v):
        s = solve_ivp(chain_rhs, [0, 4000.0], [v, 0, 0, 0], args=(KAPPA_REF, A0),
                      method="LSODA", rtol=RTOL, atol=ATOL)
        return s.y[3, -1] > 0.5
    lo, hi = 0.3, 0.9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if outcome_from_root(mid): hi = mid
        else: lo = mid
    return dict(rows=rows, all_cells_agree_with_r0_criterion=bool(agree),
                located_boundary=float(0.5 * (lo + hi)))


# ----------------------------------------------------------------------------
# 3.  Exact partial branch, kappa_c, stage-wise closed form
# ----------------------------------------------------------------------------
def branch_exact(kappa, a, n=4):
    """Exact partial equilibrium with the root held at 1."""
    disc = a * a - 4.0 * kappa
    if disc < 0:
        return None
    r = [1.0, 0.5 * (a - math.sqrt(disc))]
    for k in range(2, n):
        prev = r[-1]
        if prev <= 0.0:            # underflow far down a long weakly coupled chain
            r.append(0.0)
            continue
        fn = lambda x: x * (1 - x) * (x - a) + kappa * (prev - x)
        # fn(0) = kappa*prev > 0 and fn(a) = -kappa*(a - prev) < 0, so the low
        # root is bracketed on (0, a) whenever 0 < prev < a  (Supplementary Note 4)
        r.append(brentq(fn, 0.0, a, xtol=1e-17, rtol=8.9e-16, maxiter=300))
    return r


def branch_closed_form(kappa, a, n=4):
    r = [1.0, 0.5 * (a - math.sqrt(max(a * a - 4 * kappa, 0.0)))]
    for k in range(2, n):
        prev = r[-1]
        disc = a * a - 4 * kappa * prev
        r.append(0.5 * (a - math.sqrt(disc)))
    return r


def run_branch():
    out = {}
    out["kappa_c_a1"] = A1 ** 2 / 4
    out["kappa_c_a0"] = A0 ** 2 / 4
    out["branch_at_0.005625"] = branch_exact(0.005625, A1) is not None
    out["branch_at_0.00563"] = branch_exact(0.00563, A1) is not None
    ex = branch_exact(0.003, A1); cf = branch_closed_form(0.003, A1)
    out["profile_kappa0.003_a1"] = ex
    out["closed_form_rel_error_r2"] = (cf[2] - ex[2]) / ex[2]
    out["closed_form_rel_error_r3"] = (cf[3] - ex[3]) / ex[3]
    out["analytic_residual_stage2"] = 0.003 * cf[2] * (cf[1] - 1.0)
    out["C_on_branch_kappa0.003"] = float(np.prod(ex))
    # residual and local stability of the exact profile
    def F(r, kappa=0.003, a=A1):
        o = [0.0]
        for k in range(1, 4):
            o.append(g(r[k], a) + kappa * (r[k - 1] - r[k]))
        return np.array(o)
    out["vector_field_residual"] = float(np.max(np.abs(F(np.array(ex)))))
    pert = np.array(ex) + np.array([0.0, 1e-4, 1e-4, 1e-4])
    y = solve_ivp(lambda t, r: F(r), [0, 5000], pert, method="LSODA",
                  rtol=RTOL, atol=ATOL).y[:, -1]
    out["returns_after_1e-4_perturbation"] = float(np.max(np.abs(y - np.array(ex))))
    # propagation boundary at depleted evidence, supracritical excursion
    def propagates(kappa):
        s = solve_ivp(chain_rhs, [0, 3e4], [0.61, 0, 0, 0], args=(kappa, A0),
                      method="LSODA", rtol=RTOL, atol=ATOL)
        return s.y[3, -1] > 0.5
    lo, hi = 0.05, 0.2
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if propagates(m): hi = m
        else: lo = m
    out["propagation_boundary_a0"] = 0.5 * (lo + hi)
    return out


# ----------------------------------------------------------------------------
# 4.  Unequal coupling: min_k kappa_k, arrest depth, existence, chain length
# ----------------------------------------------------------------------------
def run_unequal():
    def final(kv, a=A0, y0=(0.61, 0, 0, 0)):
        def f(t, r):
            r = np.clip(r, 0, 1); o = np.zeros(4); o[0] = g(r[0], a)
            for k in range(1, 4):
                o[k] = g(r[k], a) + kv[k - 1] * (r[k - 1] - r[k])
            return o
        return solve_ivp(f, [0, 3e4], list(y0), method="LSODA",
                         rtol=RTOL, atol=ATOL).y[:, -1]
    bounds = []
    for stage in range(3):
        lo, hi = 0.01, 0.5
        for _ in range(60):
            m = 0.5 * (lo + hi)
            kv = [1.0, 1.0, 1.0]; kv[stage] = m
            if final(kv)[stage + 1] > 0.5: hi = m
            else: lo = m
        bounds.append(0.5 * (lo + hi))
    vectors = [[0.5, 0.05, 0.02], [0.5, 0.02, 0.05], [0.2, 0.08, 0.03],
               [0.3, 0.04, 0.07], [0.5, 0.06, 0.015], [0.2, 0.2, 0.05], [0.05, 0.2, 0.2]]
    depth_ok = True
    depths = []
    for kv in vectors:
        fin = final(kv)
        depth = int(np.sum(fin[1:] > 0.5))
        first_sub = next((i for i, k in enumerate(kv) if k <= A0 ** 2 / 4), 3)
        depths.append(dict(kappa=kv, arrest_depth=depth, first_subcritical=first_sub))
        depth_ok &= (depth == first_sub)
    # branch existence is decided by kappa_1 alone
    exists = {}
    for k1 in (0.005625, 0.00563):
        for kd in (1e-4, 0.5, 10.0, 30.0):
            exists[f"k1={k1},kd={kd}"] = (A1 ** 2 - 4 * k1) >= 0
    # kappa_c is unchanged by chain length
    lens = {}
    for n in range(2, 14):
        lo, hi = 1e-4, 0.05
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if branch_exact(m, A1, n=n) is None: hi = m
            else: lo = m
        lens[n - 1] = 0.5 * (lo + hi)
    return dict(per_stage_boundaries=bounds, arrest=depths,
                arrest_depth_matches_first_subcritical=bool(depth_ok),
                existence=exists, kappa_c_by_chain_length=lens)


# ----------------------------------------------------------------------------
# 5.  Spectrum on the branch:  lambda_1 = -sqrt(a^2 - 4 kappa) (1 - r_1)
# ----------------------------------------------------------------------------
def run_spectrum():
    worst = 0.0
    rows = []
    for a in (A1, 0.30, A0):
        for kappa in np.linspace(0.02, 0.95, 10) * (a * a / 4):
            r = branch_exact(kappa, a)
            lam_num = (1 - 2 * r[1]) * (r[1] - a) + r[1] * (1 - r[1]) - kappa
            lam_cf = -math.sqrt(a * a - 4 * kappa) * (1 - r[1])
            worst = max(worst, abs(lam_num - lam_cf))
            rows.append(dict(a=a, kappa=float(kappa), lam=float(lam_cf)))
    return dict(max_abs_discrepancy=float(worst), n_points=len(rows))


# ----------------------------------------------------------------------------
# 6.  Annihilation locus and the intermediate band
# ----------------------------------------------------------------------------
def Q_star(kappa, a0=A0, a1=A1, p=P_EXP):
    x = (a0 - 2 * math.sqrt(kappa)) / (a0 - a1)
    return x ** (1.0 / p) if 0.0 <= x <= 1.0 else float("nan")


def full_rhs(t, y, kappa, s):
    r = np.clip(y[:4], 0.0, 1.0); Q = min(max(y[4], 0.0), 1.0)
    a = a_of_Q(Q); z = r[0] - a
    o = np.zeros(5); o[0] = g(r[0], a)
    for k in range(1, 4):
        o[k] = g(r[k], a) + kappa * (r[k - 1] - r[k])
    C = float(np.prod(r))
    o[4] = EPS * (chi_plus(z, s) * C * (1 - Q) - RHO * chi_minus(z, s) * Q)
    return o


def run_band(quick=False):
    out = dict(kappa_c_a0=A0 ** 2 / 4, kappa_c_a1=A1 ** 2 / 4)
    out["Q_star"] = {k: Q_star(k) for k in (0.05, 0.08)}
    runs = {}
    if quick:                       # the full-system waits are 1e6-1e9 time units
        out["full_system"] = "skipped in --quick"
        return out
    for kappa, s, T in ((0.05, S_NARROW, 4e8), (0.08, S_NARROW, 6e6), (0.08, S_WIDE, 2e9)):
        r = branch_exact(kappa, A0)
        y0 = list(r) + [0.0]

        def fold(t, y, *a_):
            return a_of_Q(min(max(y[4], 0), 1)) ** 2 - 4 * kappa
        fold.terminal = False; fold.direction = -1

        def escape(t, y, *a_):
            return y[3] - a_of_Q(min(max(y[4], 0), 1))
        escape.terminal = True; escape.direction = 1
        sol = solve_ivp(full_rhs, [0, T], y0, args=(kappa, s), method="LSODA",
                        rtol=1e-9, atol=1e-12, events=[fold, escape], max_step=T / 50)
        t_fold = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float("nan")
        t_esc = float(sol.t_events[1][0]) if len(sol.t_events[1]) else float("nan")
        runs[f"kappa={kappa},s={s}"] = dict(
            t_fold=t_fold, t_escape=t_esc,
            lag_percent=100.0 * (t_esc - t_fold) / t_fold if t_fold == t_fold else None)
    out["full_system"] = runs
    return out


# ----------------------------------------------------------------------------
# 7.  Escape-time scaling and the zero-evidence-rate control
# ----------------------------------------------------------------------------
def escape_time(kappa, a, T=1e9):
    r0 = branch_exact(0.5 * a * a / 4, a)

    def f(t, r):
        r = np.clip(r, 0, 1); o = np.zeros(3)
        prev = 1.0
        for k in range(3):
            o[k] = g(r[k], a) + kappa * ((prev if k == 0 else r[k - 1]) - r[k])
        return o

    def ev(t, r):
        return r[2] - a
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(f, [0, T], r0[1:], method="LSODA", rtol=1e-11, atol=1e-14,
                    events=ev, max_step=T / 200)
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else float("nan")


def run_escape_scaling(quick=False):
    a = A1; kc = a * a / 4
    ex = np.array([0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3,
                   5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5])
    if quick:
        ex = ex[:10]
    t = np.array([escape_time(kc * (1 + e), a) for e in ex])
    ok = np.isfinite(t)
    lx, ly = np.log(ex[ok]), np.log(t[ok])
    local = -(np.diff(ly) / np.diff(lx))
    full = -np.polyfit(lx, ly, 1)[0]
    m = ex[ok] <= 0.05
    small = -np.polyfit(lx[m], ly[m], 1)[0] if m.sum() > 2 else float("nan")
    ctrl = {}
    for Q0 in (1.0, 0.99, 0.90, 0.70, 0.50, 0.0):
        aQ = a_of_Q(Q0); kcq = aQ * aQ / 4
        exq = np.array([0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3])
        tq = np.array([escape_time(kcq * (1 + e), aQ) for e in exq])
        good = np.isfinite(tq)
        lxq, lyq = np.log(exq[good]), np.log(tq[good])
        mq = exq[good] <= 0.05
        ctrl[Q0] = dict(a=float(aQ),
                        eps0_full=float(-np.polyfit(lxq, lyq, 1)[0]),
                        eps0_small=float(-np.polyfit(lxq[mq], lyq[mq], 1)[0]))
    return dict(excess=ex.tolist(), escape_time=t.tolist(),
                local_slope_smallest=float(local[-1]),
                fit_full_range=float(full), fit_below_5pct=float(small),
                zero_evidence_rate_control=ctrl)


# ----------------------------------------------------------------------------
# 8.  Identification from the resting profile
# ----------------------------------------------------------------------------
def run_identification(n_draws=2000, quick=False):
    if quick:
        n_draws = 400
    kappa_true, a = 0.003, A1
    r = branch_exact(kappa_true, a)
    out = dict(kappa_true=kappa_true, r=r)
    out["first_stage_exact"] = a * r[1] - r[1] ** 2
    a_two = (r[1] ** 3 - r[2] ** 2) / (r[1] ** 2 - r[2])
    out["two_ratio_a"] = a_two
    out["two_ratio_a_bias_pct"] = 100 * (a_two - a) / a
    k_two = a_two * r[1] - r[1] ** 2
    out["two_ratio_kappa_bias_pct"] = 100 * (k_two - kappa_true) / kappa_true
    for a_wrong, tag in ((0.30, "a=0.30"), (A0, "a=a0")):
        out[f"ratio_using_{tag}"] = (a_wrong * r[1] - r[1] ** 2) / kappa_true
    ah = a * 1.10
    out["ratio_10pct_threshold_error"] = (ah * r[1] - r[1] ** 2) / kappa_true
    rng = np.random.default_rng(11)
    noise = {}
    for cv in (0.01, 0.05, 0.10):
        sig = math.sqrt(math.log(1 + cv ** 2))
        r1n = r[1] * rng.lognormal(-0.5 * sig ** 2, sig, n_draws)
        r2n = r[2] * rng.lognormal(-0.5 * sig ** 2, sig, n_draws)
        k1 = a * r1n - r1n ** 2
        with np.errstate(all="ignore"):
            a2 = (r1n ** 3 - r2n ** 2) / (r1n ** 2 - r2n)
            k2 = a2 * r1n - r1n ** 2
        q = lambda v: [float(np.nanpercentile(v, p)) for p in (25, 50, 75)]
        noise[cv] = dict(first_stage=q(k1), two_ratio=q(k2))
    out["noise"] = noise
    return out


# ----------------------------------------------------------------------------
# 9.  Intermittent modulation
# ----------------------------------------------------------------------------
def modulated_escape(kbase, khigh, f_duty, period, a=A1, T=1.0):
    """Escape time from the exact partial branch under a rectangular modulation of
    kappa.  Integrated phase by phase, so the vector field is smooth on each
    interval and the discontinuity is handled exactly at the phase boundary."""
    r = np.array(branch_exact(kbase, a)[1:])

    def rhs_const(t, y, k):
        y = np.clip(y, 0.0, 1.0); o = np.zeros(3); prev = 1.0
        for j in range(3):
            o[j] = g(y[j], a) + k * ((prev if j == 0 else y[j - 1]) - y[j])
        return o

    def ev(t, y, k):
        return y[2] - a
    ev.terminal = True; ev.direction = 1

    t = 0.0
    phases = [(khigh, f_duty * period), (kbase, (1.0 - f_duty) * period)]
    guard = 0
    while t < T and guard < 200000:
        for k, dur in phases:
            if dur <= 0:
                continue
            sol = solve_ivp(rhs_const, [t, t + dur], r, args=(k,), method="LSODA",
                            rtol=1e-9, atol=1e-12, events=ev)
            if len(sol.t_events[0]):
                return float(sol.t_events[0][0])
            r = sol.y[:, -1]; t += dur
            guard += 1
            if t >= T:
                break
    return float("inf")


def run_modulation(quick=False):
    a = A1; kc = a * a / 4; kbase = 0.003
    duties = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7]
    mults = [0.5, 0.8, 1.2, 2.0, 5.0, 14.0, 30.0]
    if quick:
        duties, mults = duties[:3], mults[:4]
    t_static = escape_time(1.2 * kc, a)
    cells, agree = [], 0
    for fdc in duties:
        for m in mults:
            khigh = m * kc
            mean = fdc * khigh + (1 - fdc) * kbase
            te = modulated_escape(kbase, khigh, fdc, t_static / 50.0, a,
                                  T=60.0 * t_static)
            predicted = mean > kc
            observed = math.isfinite(te)
            agree += (predicted == observed)
            cells.append(dict(duty=fdc, mult=m, duty_weighted_mean=mean,
                              predicted=bool(predicted), observed=bool(observed)))
    return dict(static_escape_time=t_static, n_cells=len(cells),
                n_agree=int(agree), cells=cells)


# ----------------------------------------------------------------------------
# 10. Noise on the branch (Euler-Maruyama)
# ----------------------------------------------------------------------------
def run_noise(n_paths=500, T=2e4, dt=0.05, quick=False):
    if quick:
        n_paths, T = 60, 4e3
    a, kappa = A1, 0.003
    r = branch_exact(kappa, a)
    r1 = r[1]
    # barrier of the first-stage potential between r1 and the unstable partner
    ru = 0.5 * (a + math.sqrt(a * a - 4 * kappa))
    xs = np.linspace(r1, ru, 200001)
    fx = xs * (1 - xs) * (xs - a) + kappa * (1.0 - xs)
    V = -np.cumsum(fx) * (xs[1] - xs[0])
    barrier = float(V.max() - V[0])
    rng = np.random.default_rng(3)
    out = {}
    for sigma in (0.002, 0.008):
        nsteps = int(T / dt); left = 0; times = []
        y = np.tile(np.array(r[1:]), (n_paths, 1))
        alive = np.ones(n_paths, bool)
        for i in range(nsteps):
            prev = np.hstack([np.ones((n_paths, 1)), y[:, :2]])
            drift = y * (1 - y) * (y - a) + kappa * (prev - y)
            y = np.clip(y + dt * drift + sigma * math.sqrt(dt) *
                        rng.standard_normal(y.shape), 0.0, 1.0)
            esc = alive & (y[:, 0] > ru)
            if esc.any():
                times.extend([(i + 1) * dt] * int(esc.sum()))
                left += int(esc.sum()); alive &= ~esc
                y[~alive] = np.array(r[1:])
        out[sigma] = dict(n_paths=n_paths, n_escaped=left,
                          median_escape=float(np.median(times)) if times else None)
    return dict(barrier=barrier, r1=r1, r_unstable=ru, runs=out)


# ----------------------------------------------------------------------------
# 11. Supplementary Note 6: what confinement does not cover
# ----------------------------------------------------------------------------
def S_lin(x):
    return x


def S_sig(x):
    return 1.0 / (1.0 + math.exp(-(x - 0.30) / 0.05))


def additive_run(w, S, graph, y0, a=A0, T=3e4):
    """graph = 'complete' (every capacity drives every other) or 'chain'
    (capacity k receives only from k-1; the root receives nothing)."""
    def f(t, r):
        r = np.clip(r, 0.0, 1.0); o = np.zeros(4)
        for k in range(4):
            if graph == "complete":
                drive = sum(S(r[j]) for j in range(4) if j != k)
            else:
                drive = S(r[k - 1]) if k > 0 else 0.0
            o[k] = g(r[k], a) + w * (1 - r[k]) * drive
        return o
    return solve_ivp(f, [0, T], list(y0), method="LSODA",
                     rtol=RTOL, atol=ATOL).y[:, -1]


def diffusive_run(w, y0, a=A0, T=3e4, graph="complete"):
    def f(t, r):
        r = np.clip(r, 0.0, 1.0); o = np.zeros(4)
        for k in range(4):
            if graph == "complete":
                o[k] = g(r[k], a) + w * sum(r[j] - r[k] for j in range(4) if j != k)
            else:
                o[k] = g(r[k], a) + (w * (r[k - 1] - r[k]) if k > 0 else 0.0)
        return o
    return solve_ivp(f, [0, T], list(y0), method="LSODA",
                     rtol=RTOL, atol=ATOL).y[:, -1]


def bisect_weight(pred, lo, hi, n=70):
    for _ in range(n):
        m = 0.5 * (lo + hi)
        if pred(m): hi = m
        else: lo = m
    return 0.5 * (lo + hi)


def run_scope():
    out = {}
    start = [0.30] * 4
    # S8: additive coupling on the complete graph converts a sub-threshold chain
    out["additive_complete_lin_wc"] = bisect_weight(
        lambda w: additive_run(w, S_lin, "complete", start)[0] > 0.5, 0.01, 1.0)
    out["additive_complete_sig_wc"] = bisect_weight(
        lambda w: additive_run(w, S_sig, "complete", start)[0] > 0.5, 0.01, 1.0)
    out["diffusive_complete_finals"] = {
        w: np.round(diffusive_run(w, start), 6).tolist() for w in (0.1, 1.0, 10.0, 100.0)}
    # S8a: the same drive restricted to the derived chain never reaches the root
    out["additive_chain_lin_wc_r3"] = bisect_weight(
        lambda w: additive_run(w, S_lin, "chain", start)[3] > 0.5, 0.01, 5.0)
    out["additive_chain_lin_wc_r1"] = bisect_weight(
        lambda w: additive_run(w, S_lin, "chain", start)[1] > 0.5, 0.01, 5.0)
    finals = {}
    for w in (1.0, 10.0, 100.0):
        finals[w] = np.round(additive_run(w, S_lin, "chain", start), 6).tolist()
    finals["sig_from_collapsed_w100"] = np.round(
        additive_run(100.0, S_sig, "chain", [0.0] * 4), 6).tolist()
    out["additive_chain_finals"] = finals
    out["additive_chain_max_root"] = float(max(
        additive_run(w, S_lin, "chain", start)[0] for w in (0.1, 0.5, 1, 5, 20, 100)))
    out["additive_chain_C_at_w1"] = float(np.prod(np.clip(
        additive_run(1.0, S_lin, "chain", start), 0, 1)))
    # analytic checks quoted in S8
    out["symmetric_subspace_wc_lin"] = (A0 - 0.30) / 3.0
    out["symmetric_subspace_wc_sig"] = (0.30 * 0.70 * (A0 - 0.30)) / (3 * 0.70 * 0.5)
    # S9: lowering the threshold converts with no input and no coupling change
    out["threshold_sweep"] = {
        a: np.round(diffusive_run(KAPPA_REF, start, a=a, graph="chain"), 6).tolist()
        for a in (0.60, 0.40, 0.30, 0.25)}
    # S10: a crossing at the root propagates only under supercritical coupling
    out["S10"] = {k: np.round(solve_ivp(chain_rhs, [0, 3e4], [0.61, 0, 0, 0],
                                        args=(k, A0), method="LSODA",
                                        rtol=RTOL, atol=ATOL).y[:, -1], 4).tolist()
                  for k in (0.6, 0.2, 0.095, 0.085, 0.05, 0.01)}
    return out


# ----------------------------------------------------------------------------
# 12. Supplementary Note 7: the collapse criterion, the dual of kappa_c
# ----------------------------------------------------------------------------
def collapse_final(kappa, a, T=1e5):
    def f(t, r):
        r = np.clip(r, 0.0, 1.0); o = np.zeros(3); prev = 0.0
        for k in range(3):
            o[k] = g(r[k], a) + kappa * ((prev if k == 0 else r[k - 1]) - r[k])
        return o
    return solve_ivp(f, [0, T], [1.0, 1.0, 1.0], method="LSODA",
                     rtol=1e-11, atol=1e-14).y[:, -1]


# ----------------------------------------------------------------------------
# 13. Checks added for the Journal of Mathematical Biology version (v2)
#     Theorem 3 initial condition, Proposition 8(ii) passage time, and the
#     exact two-stage inversion of Proposition 9 with its conditioning.
# ----------------------------------------------------------------------------
def run_review_additions():
    out = {}
    # (a) Theorem 3 / Remark 2: at subcritical coupling a stage started above
    #     its unstable rest point recovers; started below it, it does not.
    a, kappa = A0, 0.01
    disc = a * a - 4.0 * kappa
    r_plus = 0.5 * (a + math.sqrt(disc)); r_minus = 0.5 * (a - math.sqrt(disc))
    out["remark2_kappa"] = kappa; out["remark2_r_plus"] = r_plus; out["remark2_r_minus"] = r_minus
    def final(start):
        y0 = [1.0] + [start] * 3
        s = solve_ivp(chain_rhs, [0, 2.0e4], y0, args=(kappa, a), method="LSODA",
                      rtol=1e-10, atol=1e-13)
        return np.round(s.y[:, -1], 6).tolist()
    out["remark2_final_from_0.59"] = final(0.59)   # above r_plus: stage 1 recovers, stage 2 arrests
    out["remark2_final_from_0.57"] = final(0.57)   # below r_plus: nothing recovers
    out["remark2_final_from_0.00"] = final(0.00)   # collapsed start of Theorem 3
    # (b) Proposition 8(ii): passage time of the first stage through the fold,
    #     root held at 1, from r1 = 0 to r1 = a, against pi / ((1 - a/2) sqrt(delta)).
    a = A1; kc = a * a / 4
    rows = []
    for delta in (1e-4, 1e-5, 1e-6):
        k = kc + delta
        ev = lambda t, y: y[0] - a
        ev.terminal = True
        s = solve_ivp(lambda t, y: [(1.0 - y[0]) * (y[0] ** 2 - a * y[0] + k)],
                      [0, 1e9], [0.0], events=ev, method="LSODA", rtol=1e-10, atol=1e-13)
        T = float(s.t_events[0][0]); pred = math.pi / ((1.0 - a / 2) * math.sqrt(delta))
        rows.append(dict(delta=delta, T=T, predicted=pred, ratio=T / pred, excess_time=T - pred))
    out["passage_time"] = rows
    # (c) Proposition 9: exact two-stage inversion on the true branch (equal
    #     coupling), its recovered threshold, and its sensitivity to 1% errors.
    kappa = 0.003
    r = branch_exact(kappa, a)
    r1, r2 = r[1], r[2]
    def a_two_stage(r1, r2):
        return (r1 ** 2 * (r1 - r2) - r2 ** 2 * (1 - r2)) / (r1 * (r1 - r2) - r2 * (1 - r2))
    a_rec = a_two_stage(r1, r2)
    out["two_stage_recovered_a"] = a_rec
    out["two_stage_recovered_kappa"] = a_rec * r1 - r1 ** 2
    def rel(x, ref): return (x - ref) / ref
    a_e = a_two_stage(r1, r2 * 1.01)
    out["two_stage_sens_r2_1pct_a"] = rel(a_e, a); out["two_stage_sens_r2_1pct_kappa"] = rel(a_e * r1 - r1 ** 2, kappa)
    r1e = r1 * 1.01; a_e = a_two_stage(r1e, r2)
    out["two_stage_sens_r1_1pct_a"] = rel(a_e, a); out["two_stage_sens_r1_1pct_kappa"] = rel(a_e * r1e - r1e ** 2, kappa)
    out["first_stage_sens_r1_1pct_kappa"] = rel(a * r1e - r1e ** 2, kappa)
    return out


def run_collapse():
    out = {}
    for a in (A0, A1):
        formula = (1 - a) ** 2 / 4
        b = bisect_weight(lambda k: collapse_final(k, a)[0] < 0.5, 1e-4, 1.0)
        out[a] = dict(formula=formula, bisected=b,
                      at_099=np.round(collapse_final(0.99 * formula, a), 4).tolist(),
                      at_101=np.round(collapse_final(1.01 * formula, a), 4).tolist())
    xs = np.linspace(0, 1, 2000001)
    out["max_g_over_x_at_a1"] = float(np.max((1 - xs) * (xs - A1)))
    out["window_at_a0"] = [(1 - A0) ** 2 / 4, A0 ** 2 / 4]
    out["ratio_at_a1"] = ((1 - A1) ** 2 / 4) / (A1 ** 2 / 4)
    return out



# ============================================================================
#  PART II.  Additional runs needed only by the figures
# ============================================================================

def run_confinement_examples(seed=20260814):
    """One representative trajectory per graph family, for Fig. 1b."""
    rng = np.random.default_rng(seed)
    fams = [("chain", "feedforward chain"), ("symmetric", "symmetric"),
            ("reverse", "reverse only"), ("general", "general directed")]
    t = np.linspace(0.0, 120.0, 1201)
    out = {}
    for fam, label in fams:
        n = 4
        W = draw_weights(rng, n, fam)
        r0 = rng.uniform(0.25, A0 * 0.92, size=n)

        def f(tt, r):
            r = np.clip(r, 0.0, 1.0)
            return g(r, A0) + (W @ r - W.sum(axis=1) * r)

        sol = solve_ivp(f, [0, 120.0], r0, method="LSODA", rtol=1e-9, atol=1e-12,
                        dense_output=True)
        out[label] = dict(t=t.tolist(), running_max=sol.sol(t).max(axis=0).tolist())
    return out


def run_propagation_sweep(n=41):
    """Final r1 and r3 after a supracritical excursion, against coupling (Fig. 1d)."""
    ks = np.logspace(math.log10(1e-3), math.log10(0.6), n)
    ks = np.unique(np.concatenate([ks, [0.089, 0.0895, 0.09, 0.0905, 0.091]]))
    r1, r3 = [], []
    for k in ks:
        y = solve_ivp(chain_rhs, [0, 3e4], [0.65, 0, 0, 0], args=(k, A0),
                      method="LSODA", rtol=RTOL, atol=ATOL).y[:, -1]
        r1.append(float(np.clip(y[1], 0, 1))); r3.append(float(np.clip(y[3], 0, 1)))
    return dict(kappa=ks.tolist(), r1=r1, r3=r3, kappa_c=A0 ** 2 / 4)


def run_branch_curves(n=260):
    """Exact branch and the stage-wise closed form against coupling (Fig. 2a)."""
    kc = A1 ** 2 / 4
    ks = np.linspace(1e-6, kc * 0.9995, n)
    ex, cf = [], []
    for k in ks:
        ex.append(branch_exact(k, A1)); cf.append(branch_closed_form(k, A1))
    ex, cf = np.array(ex), np.array(cf)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = 100.0 * (cf - ex) / ex
    return dict(kappa=ks.tolist(), exact=ex.tolist(), closed_form=cf.tolist(),
                percent_above=np.nan_to_num(err).tolist(), kappa_c=kc)


def run_regime_curve(n=200):
    """kappa_c(a) = a^2/4 with the evidence level read across (Fig. 2b)."""
    a = np.linspace(A1, A0, n)
    kc = a ** 2 / 4
    with np.errstate(invalid="ignore"):
        Q = ((A0 - a) / (A0 - A1)) ** (1.0 / P_EXP)
    return dict(a=a.tolist(), kappa_c=kc.tolist(), Q=np.nan_to_num(Q).tolist(),
                markers=[0.05, 0.08, 0.0056 * 1.0])


def run_stage_sweep(n=40):
    """Second-stage coupling swept with its neighbours strong (Fig. 3a)."""
    ks = np.linspace(0.02, 0.20, n)
    ks = np.unique(np.concatenate([ks, np.linspace(0.0885, 0.0915, 9)]))
    rows = []
    for k2 in ks:
        kv = [0.2, k2, 0.2]

        def f(t, r):
            r = np.clip(r, 0, 1); o = np.zeros(4); o[0] = g(r[0], A0)
            for j in range(1, 4):
                o[j] = g(r[j], A0) + kv[j - 1] * (r[j - 1] - r[j])
            return o
        y = solve_ivp(f, [0, 3e4], [0.65, 0, 0, 0], method="LSODA",
                      rtol=RTOL, atol=ATOL).y[:, -1]
        rows.append(np.clip(y[1:], 0, 1).tolist())
    return dict(kappa2=ks.tolist(), final=rows, boundary=A0 ** 2 / 4)


def run_noise_curves(sigmas=(0.002, 0.004, 0.006, 0.008), n_paths=500,
                     T=2e4, dt=0.05, quick=False):
    """Survival on the confined branch under additive noise (Fig. 3b)."""
    if quick:
        n_paths, T = 80, 6e3
    a, kappa = A1, 0.003
    r = branch_exact(kappa, a)
    ru = 0.5 * (a + math.sqrt(a * a - 4 * kappa))
    xs = np.linspace(r[1], ru, 200001)
    fx = xs * (1 - xs) * (xs - a) + kappa * (1.0 - xs)
    V = -np.cumsum(fx) * (xs[1] - xs[0])
    barrier = float(V.max() - V[0])
    rng = np.random.default_rng(3)
    nsteps = int(T / dt)
    keep = max(1, nsteps // 400)
    curves, summary = {}, {}
    for sigma in sigmas:
        y = np.tile(np.array(r[1:]), (n_paths, 1))
        alive = np.ones(n_paths, bool)
        times, frac, tgrid = [], [], []
        for i in range(nsteps):
            prev = np.hstack([np.ones((n_paths, 1)), y[:, :2]])
            drift = y * (1 - y) * (y - a) + kappa * (prev - y)
            y = np.clip(y + dt * drift + sigma * math.sqrt(dt) *
                        rng.standard_normal(y.shape), 0.0, 1.0)
            esc = alive & (y[:, 0] > ru)
            if esc.any():
                times.extend([(i + 1) * dt] * int(esc.sum()))
                alive &= ~esc
            if i % keep == 0:
                tgrid.append((i + 1) * dt); frac.append(alive.mean())
        curves[sigma] = dict(t=tgrid, fraction=frac)
        summary[sigma] = dict(n_paths=n_paths, n_escaped=n_paths - int(alive.sum()),
                              median_escape=float(np.median(times)) if times else None)
    return dict(barrier=barrier, r1=r[1], r_unstable=ru, curves=curves, runs=summary)


def run_decay_rates(n=12):
    """Decay rates on the branch against the distance from the fold (Fig. 3c)."""
    a = A1; kc = a * a / 4
    d = np.logspace(-5, math.log10(0.9), n)          # d = 1 - kappa/kappa_c
    rows = []
    for dd in d:
        kappa = kc * (1 - dd)
        r = branch_exact(kappa, a)
        lam1 = (1 - 2 * r[1]) * (r[1] - a) + r[1] * (1 - r[1]) - kappa
        lam2 = (1 - 2 * r[2]) * (r[2] - a) + r[2] * (1 - r[2]) - kappa
        rows.append(dict(d=float(dd), kappa=float(kappa),
                         minus_lam1=float(-lam1), minus_lam2=float(-lam2),
                         closed_form=float(math.sqrt(a * a - 4 * kappa) * (1 - r[1]))))
    return rows


def run_period_sweep(n=13, quick=False):
    """Modulated escape time against the modulation period (Fig. 4b)."""
    a = A1; kc = a * a / 4; kbase = 0.003
    t_static = escape_time(1.2 * kc, a)
    f_duty = 0.5
    khigh = (1.2 * kc - (1 - f_duty) * kbase) / f_duty
    periods = np.logspace(-3, math.log10(5.0), n) * t_static
    rows = []
    for per in periods:
        te = modulated_escape(kbase, khigh, f_duty, per, a, T=80.0 * t_static)
        rows.append(dict(period_over_static=float(per / t_static),
                         ratio=float(te / t_static) if math.isfinite(te) else None,
                         inside_first_phase=bool(math.isfinite(te) and te <= f_duty * per)))
    return dict(t_static=t_static, kappa_high=khigh, duty=f_duty, rows=rows)


def full_escape_time(kappa, Q0, eps, s=S_NARROW, T=5e7):
    """Escape time in the full five-dimensional system from the exact branch."""
    a0q = a_of_Q(Q0)
    r = branch_exact(0.5 * a0q * a0q / 4, a0q)
    if r is None:
        return float("nan")
    y0 = list(r) + [Q0]

    def rhs(t, y):
        rr = np.clip(y[:4], 0.0, 1.0); Q = min(max(y[4], 0.0), 1.0)
        a = a_of_Q(Q); z = rr[0] - a
        o = np.zeros(5); o[0] = g(rr[0], a)
        for k in range(1, 4):
            o[k] = g(rr[k], a) + kappa * (rr[k - 1] - rr[k])
        o[4] = eps * (chi_plus(z, s) * float(np.prod(rr)) * (1 - Q)
                      - RHO * chi_minus(z, s) * Q)
        return o

    def ev(t, y):
        return y[3] - a_of_Q(min(max(y[4], 0), 1))
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, [0, T], y0, method="LSODA", rtol=1e-10, atol=1e-13,
                    events=ev, max_step=T / 400)
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else float("nan")


def run_full_exponents(quick=False):
    """Exponent fitted below 5% excess in the full system, evidence on and off
    (Fig. 4c inset and Supplementary Note 2)."""
    Q0s = (1.0, 0.99, 0.90, 0.70, 0.50, 0.0)
    ex = np.array([0.005, 0.01, 0.02, 0.05] if quick else
                  [0.005, 0.0075, 0.01, 0.02, 0.035, 0.05])
    out = {}
    for eps in (0.02, 0.0):
        rows = {}
        for Q0 in Q0s:
            a = a_of_Q(Q0); kc = a * a / 4
            t = np.array([full_escape_time(kc * (1 + e), Q0, eps) for e in ex])
            ok = np.isfinite(t)
            if ok.sum() < 3:
                rows[Q0] = dict(a=float(a), exponent=float("nan")); continue
            slope = -np.polyfit(np.log(ex[ok]), np.log(t[ok]), 1)[0]
            rows[Q0] = dict(a=float(a), exponent=float(slope))
        out[eps] = rows
    return out


def run_staircase(excess=0.05):
    """Trajectory of a coupling-driven escape at 5% excess (Fig. 4d)."""
    a = A1; kc = a * a / 4; kappa = kc * (1 + excess)
    r0 = branch_exact(0.5 * kc, a)

    def f(t, r):
        r = np.clip(r, 0, 1); o = np.zeros(3); prev = 1.0
        for k in range(3):
            o[k] = g(r[k], a) + kappa * ((prev if k == 0 else r[k - 1]) - r[k])
        return o

    def ev(t, r):
        return r[2] - a
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(f, [0, 1e7], r0[1:], method="LSODA", rtol=1e-11, atol=1e-14,
                    events=ev, dense_output=True, max_step=1e4)
    T = float(sol.t_events[0][0])
    t = np.linspace(0, T, 4001)
    Y = sol.sol(t)
    cross = []
    for k in range(3):
        i = np.argmax(Y[k] >= a)
        j = np.argmax(Y[k] >= 0.9)
        cross.append(dict(stage=k + 1, cross_fraction=float(t[i] / T),
                          reach09_fraction=float(t[j] / T) if Y[k].max() >= 0.9 else None))
    return dict(escape_time=T, t_over_T=(t / T).tolist(), Y=Y.tolist(),
                a=a, crossings=cross)


# ============================================================================
#  PART III.  Figures 1 to 4
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

# Okabe-Ito derived, colour-vision-deficiency safe; every series also carries a
# distinct dash pattern or marker, so identity is never colour alone.
BLUE, GREEN, ORANGE, PINK, INK, MUTED = ("#0173B2", "#029E73", "#D55E00",
                                         "#CC78BC", "#222222", "#666666")
SERIES = [(BLUE, "-"), (GREEN, "--"), (ORANGE, ":"), (PINK, "-.")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "dejavusans",
})


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


def fig1(R, path="Fig1.png"):
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.09, right=0.97, top=0.94, bottom=0.09)

    # ---- a: the chain -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    panel_label(ax, "a", dx=-0.02, dy=1.02)
    names = [r"$r_0$", r"$r_1$", r"$r_2$", r"$r_3$"]
    sub = ["sensory", "policy", "motiv.", "fast vol."]
    x0, y0, w, h, gap = 0.06, 0.50, 0.16, 0.17, 0.09
    for i in range(4):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y0), w, h, fill=False, lw=1.4,
                               edgecolor=INK, transform=ax.transAxes))
        ax.text(x + w / 2, y0 + h / 2, names[i], transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.text(x + w / 2, y0 - 0.07, sub[i], transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=MUTED)
        if i < 3:
            ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w, y0 + h / 2),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
            ax.text(x + w + gap / 2, y0 + h / 2 + 0.055, r"$\kappa$",
                    transform=ax.transAxes, ha="center", fontsize=10, color=GREEN)
    ax.annotate("", xy=(x0 + w / 2, y0 + h), xytext=(x0 + w / 2, y0 + h + 0.20),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8))
    ax.text(x0 + w / 2 + 0.03, y0 + h + 0.21, r"$u(t)$  root excursion",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=ORANGE)
    ax.text(0.5, 0.28, r"crossing set by $u$   |   propagation set by $\kappa$",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK)
    ax.text(0.5, 0.15,
            "while every capacity is below threshold no non-negative\n"
            "coupling can cross, on this or any other graph",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTED, style="italic")

    # ---- b: running maximum -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    for (label, d), (c, ls) in zip(R["confinement_examples"].items(), SERIES):
        ax.plot(d["t"], d["running_max"], color=c, ls=ls, label=label)
    ax.axhline(A0, color=INK, ls=":", lw=1.2)
    ax.text(0.02, A0 + 0.015, r"threshold $a_0$", transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK)
    ax.set_xlabel("time"); ax.set_ylabel(r"$\max_k r_k(t)$")
    ax.set_ylim(-0.02, 0.75); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    c = R["confinement"]
    ax.text(0.97, 0.06, f"{c['reached_threshold']} of {c['n_systems']} runs reached "
            r"$a_0$;" "\n" f"the maximum never rose "
            f"({c['largest_rise_of_running_max']:.0e})",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK)

    # ---- c: pulse grid ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c")
    taus = [row["tau"] for row in R["pulse"]["rows"]]
    for i, row in enumerate(R["pulse"]["rows"]):
        for cell in row["cells"]:
            ax.plot(cell["r0_at_tau"], i,
                    marker="o" if cell["converted"] else "x",
                    color=BLUE if cell["converted"] else ORANGE,
                    ms=6 if cell["converted"] else 7, mew=1.8, ls="none")
    ax.axvline(A0, color=INK, ls=":", lw=1.2)
    ax.text(A0 - 0.008, 2.6, r"$a_0 = 0.60$", rotation=90, ha="right", va="center",
            fontsize=8)
    ax.set_yticks(range(len(taus)))
    ax.set_yticklabels([rf"$\tau = {t:g}$" for t in taus])
    ax.set_ylim(-0.8, len(taus) - 0.4)
    ax.set_xlabel(r"root value at the end of the pulse,  $r_0(\tau)$")
    ax.set_ylabel("pulse duration")
    ax.plot([], [], "o", color=BLUE, ms=6, label="recovers")
    ax.plot([], [], "x", color=ORANGE, ms=7, mew=1.8, label="collapses")
    ax.legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=1.0)

    # ---- d: propagation is set by the coupling ------------------------------
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d")
    s = R["propagation_sweep"]
    ax.plot(s["kappa"], s["r3"], color=BLUE, ls="-", label=r"$r_3$")
    ax.plot(s["kappa"], s["r1"], color=ORANGE, ls="--", marker="o", ms=3,
            label=r"$r_1$")
    ax.axvline(s["kappa_c"], color=INK, ls=":", lw=1.2)
    ax.text(s["kappa_c"] * 0.93, 0.45, r"$\kappa_c = a_0^2/4$", rotation=90,
            ha="right", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel(r"coupling  $\kappa$")
    ax.set_ylabel("final state after the crossing")
    ax.text(0.03, 0.97, "root crossed at $v = 0.65$ in every run",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
    ax.text(0.03, 0.88, f"boundary located at "
            f"{R['branch']['propagation_boundary_a0']:.7f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
    ax.text(0.965, 0.93, r"$r_3$", transform=ax.transAxes, color=BLUE, fontsize=9,
            ha="right", va="center")
    ax.text(0.05, 0.13, r"$r_1$", transform=ax.transAxes, color=ORANGE, fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig2(R, path="Fig2.png"):
    fig = plt.figure(figsize=(9.6, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.98,
                          top=0.88, bottom=0.19)

    # ---- a: the branch ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.24)
    b = R["branch_curves"]
    ks = np.array(b["kappa"]); ex = np.array(b["exact"]); cf = np.array(b["closed_form"])
    for j, (c, ls) in zip((1, 2, 3), SERIES[:3]):
        ax.plot(ks, ex[:, j], color=c, ls=ls, label=rf"$r_{j}$")
        ax.plot(ks, cf[:, j], color="0.55", lw=1.0, ls="-", zorder=1)
    ax.axvline(b["kappa_c"], color=INK, ls=":", lw=1.2)
    ax.text(b["kappa_c"] * 0.97, 2e-11, r"$\kappa_c = a_1^2/4$", rotation=90,
            ha="right", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-12, 5e-1)
    ax.set_xlabel(r"coupling  $\kappa$"); ax.set_ylabel("partial-branch capacity")
    for j, c in zip((1, 2, 3), (BLUE, GREEN, ORANGE)):
        ax.text(1.005, ex[-1, j], rf"$r_{j}$", transform=ax.get_yaxis_transform(),
                color=c, fontsize=9, va="center")
    ax.text(0.30, 0.045, "colour: exact equilibrium\ngrey: stage-wise closed form",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    axi = ax.inset_axes([0.44, 0.22, 0.44, 0.24])
    err = np.array(b["percent_above"])
    for j, (c, ls) in zip((1, 2, 3), SERIES[:3]):
        axi.plot(ks, err[:, j], color=c, ls=ls, lw=1.2)
    axi.set_title("% above exact", fontsize=7, pad=2)
    axi.tick_params(labelsize=6); axi.set_xticks([])
    axi.text(0.03, 0.16, r"$r_1$: 0", transform=axi.transAxes, fontsize=6, color=BLUE)

    # ---- b: three regimes ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.24)
    c2 = R["regime_curve"]
    a = np.array(c2["a"]); kc = np.array(c2["kappa_c"])
    ax.fill_between(a, A0 ** 2 / 4, 0.115, color=BLUE, alpha=0.12, lw=0)
    ax.fill_between(a, A1 ** 2 / 4, A0 ** 2 / 4, color=GREEN, alpha=0.12, lw=0)
    ax.fill_between(a, 0.0, A1 ** 2 / 4, color=ORANGE, alpha=0.12, lw=0)
    ax.plot(a, kc, color=INK, lw=1.8)
    for k in (0.05, 0.08):
        ax.plot(2 * math.sqrt(k), k, "o", color=INK, ms=5)
    ax.set_xlim(A1, A0); ax.set_ylim(0, 0.115)
    ax.set_yticks([0, A1 ** 2 / 4, 0.05, A0 ** 2 / 4])
    ax.set_yticklabels(["0", "0.0056", "0.050", "0.090"])
    ax.set_xlabel(r"threshold  $a(Q)$"); ax.set_ylabel(r"coupling  $\kappa$")
    ax.text(0.50, 0.93, "propagates at once", transform=ax.transAxes, ha="center",
            fontsize=8)
    ax.text(0.16, 0.55, "stalls, completes later", transform=ax.transAxes, fontsize=8)
    ax.text(0.55, 0.035, "confined to the branch", transform=ax.transAxes, fontsize=8)
    ax.annotate(r"$\kappa_c(a) = a^2/4$,  read across: $Q^{*}(\kappa)$",
                xy=(2 * math.sqrt(0.05), 0.05), xytext=(0.20, 0.30),
                textcoords="axes fraction", fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    axt = ax.secondary_xaxis("top", functions=(
        lambda x: ((A0 - np.clip(x, A1, A0)) / (A0 - A1)) ** (1 / P_EXP),
        lambda q: A0 - (A0 - A1) * np.clip(q, 0, 1) ** P_EXP))
    axt.set_xlabel("evidence  $Q$", fontsize=9); axt.tick_params(labelsize=8)
    axt.set_xticks([0.95, 0.8, 0.6, 0.4, 0.0])

    # ---- c: identification --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.24)
    ident = R["identification"]; kt = ident["kappa_true"]
    xs = [0, 1, 2, 3]; labels = ["0", "1%", "5%", "10%"]
    fs_med = [ident["first_stage_exact"] / kt]
    fs_lo, fs_hi = [ident["first_stage_exact"] / kt], [ident["first_stage_exact"] / kt]
    tr_med = [(1 + ident["two_ratio_kappa_bias_pct"] / 100)]
    tr_lo, tr_hi = list(tr_med), list(tr_med)
    for cv in (0.01, 0.05, 0.10):
        q = ident["noise"][cv] if cv in ident["noise"] else ident["noise"][str(cv)]
        fs_lo.append(q["first_stage"][0] / kt); fs_med.append(q["first_stage"][1] / kt)
        fs_hi.append(q["first_stage"][2] / kt)
        tr_lo.append(q["two_ratio"][0] / kt); tr_med.append(q["two_ratio"][1] / kt)
        tr_hi.append(q["two_ratio"][2] / kt)
    off = 0.10
    ax.errorbar(np.array(xs) - off, tr_med,
                yerr=[np.array(tr_med) - np.array(tr_lo),
                      np.array(tr_hi) - np.array(tr_med)],
                fmt="o", color=ORANGE, ms=5, capsize=0, lw=1.4,
                label="two stage ratios")
    ax.errorbar(np.array(xs) + off, fs_med,
                yerr=[np.array(fs_med) - np.array(fs_lo),
                      np.array(fs_hi) - np.array(fs_med)],
                fmt="s", color=BLUE, ms=5, capsize=0, lw=1.4,
                label=r"$\kappa = a r_1 - r_1^2$")
    ax.axhline(1.0, color=INK, ls=":", lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("lognormal noise on the stage ratios")
    ax.set_ylabel(r"$\hat{\kappa}$ / true value")
    ax.text(0.03, 0.97, "evaluated on the exact equilibrium", transform=ax.transAxes,
            va="top", fontsize=8, color=MUTED)
    ax.legend(loc="lower left")
    fig.savefig(path); plt.close(fig)
    return path


def fig3(R, path="Fig3.png"):
    fig = plt.figure(figsize=(9.6, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.36, left=0.07, right=0.98,
                          top=0.88, bottom=0.20)

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.22)
    s = R["stage_sweep"]; F = np.array(s["final"])
    for j, (c, ls) in enumerate(SERIES[:3]):
        ax.plot(s["kappa2"], F[:, j], color=c, ls=ls, label=rf"$r_{j+1}$")
    ax.axvline(s["boundary"], color=INK, ls=":", lw=1.2)
    ax.text(s["boundary"] * 1.02, 0.55, r"$a_0^2/4$", fontsize=8)
    ax.set_xlabel(r"coupling of the second stage  $\kappa_2$")
    ax.set_ylabel("final capacity")
    ax.text(0.03, 0.62, r"$\kappa_1 = \kappa_3 = 0.2$", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.35))

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.22)
    nc = R["noise_curves"]
    for (sig, d), (c, ls) in zip(sorted(nc["curves"].items(),
                                        key=lambda kv: float(kv[0])), SERIES):
        ax.plot(d["t"], d["fraction"], color=c, ls=ls,
                label=rf"$\sigma = {float(sig):g}$")
    ax.set_xlabel("time"); ax.set_ylabel("fraction still on the branch")
    ax.set_ylim(-0.02, 1.08); ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.14))
    ax.text(0.03, 0.06, f"barrier {nc['barrier']:.1e} at "
            r"$\kappa = 0.003$", transform=ax.transAxes, fontsize=8, color=MUTED)

    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.22)
    rows = R["decay_rates"]
    d = [r["d"] for r in rows]
    ax.plot(d, [r["closed_form"] for r in rows], color=INK, lw=1.4,
            label=r"$\sqrt{a^2-4\kappa}\,(1-r_1)$", zorder=1)
    ax.plot(d, [r["minus_lam1"] for r in rows], "o", color=BLUE, ms=5,
            label=r"$-\lambda_1$, numerical", zorder=2)
    ax.plot(d, [r["minus_lam2"] for r in rows], "s", color=GREEN, ms=5,
            label=r"$-\lambda_2$", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"distance from the fold  $1 - \kappa/\kappa_c$")
    ax.set_ylabel("decay rate on the branch")
    ax.text(0.06, 0.55, "slope 1/2", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="lower right")
    fig.savefig(path); plt.close(fig)
    return path


def fig4(R, path="Fig4.png"):
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                          left=0.09, right=0.97, top=0.94, bottom=0.09)
    kc = A1 ** 2 / 4; kbase = 0.003

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    for cell in R["modulation"]["cells"]:
        ax.plot(cell["duty"], cell["mult"] * kc,
                marker="o" if cell["observed"] else "x",
                color=BLUE if cell["observed"] else ORANGE,
                ms=6 if cell["observed"] else 7, mew=1.8, ls="none")
    f = np.linspace(0.03, 0.78, 200)
    ax.plot(f, (kc - (1 - f) * kbase) / f, color=INK, lw=1.5)
    ax.set_yscale("log"); ax.set_xlabel(r"duty fraction  $f$")
    ax.set_ylabel(r"elevated coupling  $\kappa_{\mathrm{high}}$")
    ax.text(0.42, 0.80, r"$f\kappa_{\mathrm{high}} + (1-f)\kappa_{\mathrm{low}} = \kappa_c$",
            transform=ax.transAxes, fontsize=9)
    ax.plot([], [], "o", color=BLUE, ms=6, label="escape")
    ax.plot([], [], "x", color=ORANGE, ms=7, mew=1.8, label="no escape")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.3, columnspacing=1.2)

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    ps = R["period_sweep"]
    xs = [r["period_over_static"] for r in ps["rows"] if r["ratio"]]
    ys = [r["ratio"] for r in ps["rows"] if r["ratio"]]
    ax.plot(xs, ys, color=BLUE, marker="o", ms=4)
    for r in ps["rows"]:
        if r["ratio"] and r["inside_first_phase"]:
            ax.plot(r["period_over_static"], r["ratio"], "D", color=ORANGE, ms=6)
    ax.axhline(1.0, color=INK, ls=":", lw=1.2)
    ax.text(0.02, 1.04, "averaging prediction", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("modulation period / static escape time")
    ax.set_ylabel("escape time / static escape time")
    ax.text(0.03, 0.28, "diamonds: escape completes\nwithin the first elevated phase",
            transform=ax.transAxes, fontsize=8, color=ORANGE)
    ax.text(0.03, 0.10, "below the line: the modulated system escapes\n"
            "sooner than the average predicts",
            transform=ax.transAxes, fontsize=8, color=INK)

    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c")
    e = R["escape"]; ex = np.array(e["excess"]); t = np.array(e["escape_time"])
    ok = np.isfinite(t); lx, ly = np.log(ex[ok]), np.log(t[ok])
    local = -(np.diff(ly) / np.diff(lx))
    mid = np.sqrt(ex[ok][1:] * ex[ok][:-1])
    ax.plot(mid, local, color=BLUE, marker="o", ms=4)
    ax.axhline(0.5, color=INK, ls=":", lw=1.2)
    ax.text(mid.min() * 1.15, 0.503, "1/2", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"excess  $(\kappa - \kappa_c)/\kappa_c$")
    ax.set_ylabel("local slope of  log T  vs  log excess")
    ax.text(0.05, 0.20, f"slope {e['local_slope_smallest']:.4f} at the smallest excess",
            transform=ax.transAxes, fontsize=8)
    axi = ax.inset_axes([0.16, 0.62, 0.52, 0.34])
    fe = R["full_exponents"]
    for eps, c, mk, lab in ((0.0, "0.35", "o", r"$\varepsilon = 0$"),
                            (0.02, ORANGE, "s", r"$\varepsilon = 0.02$")):
        rows = fe[eps] if eps in fe else fe[str(eps)]
        aa = [v["a"] for v in rows.values()]; yy = [v["exponent"] for v in rows.values()]
        o = np.argsort(aa)
        axi.plot(np.array(aa)[o], np.array(yy)[o], marker=mk, ms=4, color=c,
                 lw=1.2, label=lab)
    axi.set_title("exponent fitted below 5% excess", fontsize=7, pad=2)
    axi.set_xlabel(r"threshold  $a(Q_0)$", fontsize=7)
    axi.tick_params(labelsize=6)
    axi.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.02),
               handlelength=1.6, borderaxespad=0.2)

    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d")
    st = R["staircase"]; Y = np.array(st["Y"])
    lab_y = [1.0, 0.90, None]
    for j, (c, ls) in enumerate(SERIES[:3]):
        ax.plot(st["t_over_T"], Y[j], color=c, ls=ls)
        ax.text(1.02, lab_y[j] if lab_y[j] is not None else Y[j][-1],
                rf"$r_{j+1}$", transform=ax.get_yaxis_transform(),
                color=c, fontsize=9, va="center")
    ax.axhline(st["a"], color=INK, ls=":", lw=1.0)
    ax.text(0.01, st["a"] + 0.015, r"$a_1$", transform=ax.get_yaxis_transform(),
            fontsize=8)
    ax.set_xlabel("time / escape time"); ax.set_ylabel("capacity")
    c1, c2 = st["crossings"][0], st["crossings"][1]
    ax.text(0.03, 0.99, r"$r_1$ and $r_2$ cross $a_1$ at "
            f"{c1['cross_fraction']:.2f} and {c2['cross_fraction']:.2f}"
            f"\nof the escape interval",
            transform=ax.transAxes, va="top", fontsize=8.5)
    ax.text(0.03, 0.80, r"$r_3$ crosses at 1.00 by construction",
            transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
    ax.set_ylim(-0.04, 1.16)
    fig.savefig(path); plt.close(fig)
    return path


# ============================================================================
# 13.  Secondary analysis of published aggregate data (Results, Fig. 5)
# ----------------------------------------------------------------------------
# Sources, both open access, no individual patient records:
#   ref. 23  Bot L. et al. Acta Psychiatr Scand 154, 99-115 (2026), Tables 1, 2
#   ref. 11  Pelzer A.C.M. et al. Neuropsychiatr Dis Treat 14, 317-326 (2018)
# ----------------------------------------------------------------------------
BOT_T2 = [  # label, N, BFCRS pre, SD pre, BFCRS post, mean dose (mg)
    ("Bush 1996 G1", 16, 17.7, 6.8, 2.07, 3.0), ("Bush 1996 G2", 5, 17.4, 3.5, 13.08, 5.5),
    ("Johnson 2022 C3", 1, 13, None, 13, 8.0), ("Lee 2000", 24, 15.52, 3.72, 2.48, 6.0),
    ("Lin and Huang 2013 G1", 13, 8.9, 2.8, 0, 2.0), ("Lin and Huang 2013 G2", 3, 11.7, 2.5, 0, 4.0),
    ("Lin and Huang 2013 G3", 2, 11.5, 3.5, 0, 8.25), ("Lin and Huang 2013 G4", 3, 11.7, 3.1, 0, 8.25),
    ("Miles 2019 C5", 1, 26, None, 0, 14.0), ("Sakhardande 2022 C1", 1, 22, None, 17, 6.0),
    ("Sakhardande 2022 C2", 1, 10, None, 0, 6.0), ("Sakhardande 2022 C3", 1, 8, None, 0, 4.0),
    ("Seetharaman 2021 C1", 1, 18, None, 0, 8.0), ("Seetharaman 2021 C2", 1, 27, None, 4, 10.0),
    ("Seetharaman 2021 C3", 1, 32, None, 6, 10.0), ("Sharma 2017", 5, 16.3, 4.5, 0, 2.2),
    ("Suchandra 2021 2mg", 37, 12, None, 6, 2.0), ("Suchandra 2021 4mg", 20, 15, None, 7.5, 4.0),
    ("Thamizh 2016 C2", 1, 12, None, 10, 8.0), ("Thamizh 2016 C4", 1, 9, None, 0, 8.0)]

BOT_T1 = [  # study, N, highest daily lorazepam-equivalent dose, response %, remission %
    ("Aloysi 2011",4,6,100,0), ("Appiani 2023",22,2,27,None), ("Benazzi 1991",4,14,100,None),
    ("Bhattacharjee 2023",3,14,0,0), ("Brelinski 2009",3,15,100,None), ("Bruijn and Blom 2010",3,60,100,100),
    ("Bush 1996",21,8,76,52), ("Cottencin 2007",12,20,100,83), ("Dutt 2011",51,12,18,None),
    ("England 2011",25,16,96,28), ("Fricchione 1983",4,24,100,100), ("Hatta 2007",41,13.6,49,2),
    ("Huang 2005",14,7.2,100,None), ("Huang 2013",12,7.2,100,100), ("Hung and Huang 2006",7,7.2,100,100),
    ("Jaimes-Albornoz 2013",7,10,71,43), ("Johnson 2022",3,14,0,0), ("Kritzinger 2001",9,2,67,None),
    ("Layek 2022",66,16,82,None), ("Lee 2000",24,16,79,0), ("Lin and Huang 2013",21,8.25,100,None),
    ("Lin 2016",30,8.25,84,79), ("Lin 2017",21,8.25,90,86), ("Martenyi 1989",4,6,100,None),
    ("Miles 2019",7,20,100,14), ("Mishra 2023",3,2,100,100), ("Nahar 2017",36,8,75,50),
    ("Narayanaswamy 2012",99,8,32,32), ("Northoff 1995",18,10,67,None), ("Northoff 1998",22,4,59,59),
    ("Payee 1999",30,8,70,70), ("Ramdurg 2013",61,12,67,None), ("Rosebush 1990",15,2,87,80),
    ("Rosebush and Mazurek 1996",5,3,100,None), ("Sakhardande 2022",5,8,100,40), ("Salam 1987",3,5,100,100),
    ("Salam and Kilzieh 1988",5,6,60,20), ("Schmider 1999",17,6,59,None), ("Seethalakshmi 2008",16,4,100,75),
    ("Seetharaman 2021",4,12,100,33), ("Sharma 2017",5,2,100,100), ("Smith 2015",4,6.7,100,100),
    ("Suchandra 2021",57,4,68,0), ("Thamizh 2016",4,24,50,50), ("Tibrewal 2010",99,6,69,32),
    ("Tuerlings 2010",26,12,21,13), ("Unal 2017",55,15,24,24), ("Ungvari 1994",18,6,100,22),
    ("Ungvari 1999",18,6,0,0), ("Wetzel and Benkert 1988",10,5,100,None), ("White 2015",3,18,67,33),
    ("Wilson 2015",232,5.8,96,85), ("Yassa 1990",10,6,100,100)]

PELZER = [("Lee 1997","benzodiazepine",73,27), ("Northoff","benzodiazepine",80,6.6),
          ("Seethalakshmi","benzodiazepine",75,18), ("Tibrewal","benzodiazepine",32.3,68.7),
          ("Cristancho","ECT",80,20), ("Raveendranathan","ECT",88.89,11.11)]

MID_LO, MID_HI, P_GRADED, MIN_N = 35.0, 65.0, 0.28, 10


def run_published_data():
    """Depth of recovery against dose, and the case-level partial window."""
    from scipy import stats
    ind = sorted([(l, pre, post, 100.0 * (pre - post) / pre)
                  for l, N, pre, sd, post, d in BOT_T2 if N == 1], key=lambda r: r[3])
    red = np.array([r[3] for r in ind]); n = len(red)
    mid = int(((red > MID_LO) & (red < MID_HI)).sum())
    srt = np.sort(red); j = int(np.argmax(np.diff(srt)))
    rows = [(s, N, d, r, m, m / r) for s, N, d, r, m in BOT_T1
            if r not in (None, 0) and m is not None and N >= MIN_N]
    D = np.array([x[2] for x in rows]); Qd = np.array([x[5] for x in rows])
    Rr = np.array([x[3] for x in rows], float)
    lr = stats.linregress(np.log(D), Qd); sp = stats.spearmanr(D, Qd)
    lr2 = stats.linregress(np.log(D), Rr)
    share = np.array([p / (r + p) for _, _, r, p in PELZER])
    return dict(
        cases=[dict(label=l, pre=pre, post=post, reduction=pct) for l, pre, post, pct in ind],
        n_cases=n, n_in_window=mid,
        gap_lo=float(srt[j]), gap_hi=float(srt[j + 1]), gap=float(np.diff(srt)[j]),
        p_binomial=float(stats.binomtest(mid, n, P_GRADED, alternative="less").pvalue),
        bias_curve=[(f, float(P_GRADED * f / (P_GRADED * f + 1 - P_GRADED)),
                     float(stats.binom.pmf(0, n, P_GRADED * f / (P_GRADED * f + 1 - P_GRADED))))
                    for f in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)],
        studies=[dict(study=s, n=N, dose=d, response=r, remission=m, depth=q) for s, N, d, r, m, q in rows],
        k_studies=len(rows), dose_lo=float(D.min()), dose_hi=float(D.max()),
        depth_lo=float(Qd.min()), depth_hi=float(Qd.max()),
        depth_slope=float(lr.slope), depth_p=float(lr.pvalue),
        depth_rho=float(sp.statistic), depth_rho_p=float(sp.pvalue),
        response_slope=float(lr2.slope), response_p=float(lr2.pvalue),
        partial_share_median=float(np.median(share)),
        partial_share_lo=float(share.min()), partial_share_hi=float(share.max()))





def fig5(R, path="Fig5.png"):
    """Published data: depth against dose, the case-level window, and its bound."""
    from scipy import stats
    P = R["published"]
    fig = plt.figure(figsize=(9.6, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.36, left=0.07, right=0.98,
                          top=0.88, bottom=0.20)

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.22)
    d = np.array([s["dose"] for s in P["studies"]])
    q = np.array([s["depth"] for s in P["studies"]])
    nn = np.array([s["n"] for s in P["studies"]], float)
    ax.scatter(d, q, s=12 + 90 * nn / nn.max(), color=BLUE, alpha=0.75,
               edgecolors="none", zorder=3)
    lx = np.log(d); xs = np.linspace(d.min(), d.max(), 120)
    lr = stats.linregress(lx, q)
    fit = lr.intercept + lr.slope * np.log(xs)
    s2 = np.sum((q - (lr.intercept + lr.slope * lx)) ** 2) / (len(d) - 2)
    se = np.sqrt(s2 * (1.0 / len(d) + (np.log(xs) - lx.mean()) ** 2
                       / np.sum((lx - lx.mean()) ** 2)))
    ax.fill_between(xs, fit - 1.96 * se, fit + 1.96 * se, color=INK, alpha=0.12,
                    lw=0, zorder=1)
    ax.plot(xs, fit, color=INK, lw=1.4, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("highest daily dose (mg lorazepam-equivalent)")
    ax.set_ylabel("depth of recovery")
    ax.set_ylim(-0.08, 1.10)
    ax.text(0.03, 0.06, rf"$P = {P['depth_p']:.2f}$", transform=ax.transAxes, fontsize=8)

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.22)
    red = [c["reduction"] for c in P["cases"]]
    ax.axhspan(MID_LO, MID_HI, color=ORANGE, alpha=0.15, lw=0, zorder=1)
    ax.plot(np.arange(1, len(red) + 1), red, "o", color=BLUE, ms=6, zorder=3)
    ax.set_xlabel("individually reported patient")
    ax.set_ylabel("BFCRS reduction (%)")
    ax.set_ylim(-8, 110); ax.set_xticks(range(1, len(red) + 1))
    ax.text(0.04, 0.44, "window empty\n" + rf"$P = {P['p_binomial']:.3f}$",
            transform=ax.transAxes, fontsize=8)

    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.22)
    f = [b[0] for b in P["bias_curve"]]; pr = [b[2] for b in P["bias_curve"]]
    ax.plot(f, pr, "-o", color=GREEN, ms=5)
    ax.axvline(0.2, color=INK, ls=":", lw=1.2)
    ax.text(0.215, 0.10, "fivefold\nsuppression", fontsize=8, color=MUTED)
    ax.set_xlabel("relative publication rate,\npartial versus other cases")
    ax.set_ylabel(r"$P$(no patient in the window)")
    ax.set_ylim(-0.04, 1.04)
    fig.savefig(path); plt.close(fig)
    return path


# ============================================================================
#  PART IV.  Source-data workbook, one sheet per panel
# ============================================================================
def write_source_data(R, path="two_axes_source_data.xlsx"):
    from openpyxl import Workbook
    wb = Workbook(); wb.remove(wb.active)

    def sheet(name, header, rows):
        ws = wb.create_sheet(name[:31]); ws.append(header)
        for r in rows:
            ws.append(list(r))

    ce = R["confinement_examples"]
    keys = list(ce.keys())
    sheet("Fig1b", ["time"] + keys,
          zip(ce[keys[0]]["t"], *[ce[k]["running_max"] for k in keys]))
    sheet("Fig1c", ["tau", "amplitude_multiple", "r0_at_tau", "converted"],
          [(row["tau"], c["mult"], c["r0_at_tau"], c["converted"])
           for row in R["pulse"]["rows"] for c in row["cells"]])
    s = R["propagation_sweep"]
    sheet("Fig1d", ["kappa", "r1_final", "r3_final"], zip(s["kappa"], s["r1"], s["r3"]))
    b = R["branch_curves"]
    sheet("Fig2a", ["kappa", "r1_exact", "r2_exact", "r3_exact",
                    "r1_closed", "r2_closed", "r3_closed",
                    "pct_above_r2", "pct_above_r3"],
          [(k, e[1], e[2], e[3], c[1], c[2], c[3], p[2], p[3])
           for k, e, c, p in zip(b["kappa"], b["exact"], b["closed_form"],
                                 b["percent_above"])])
    c2 = R["regime_curve"]
    sheet("Fig2b", ["a", "kappa_c", "Q"], zip(c2["a"], c2["kappa_c"], c2["Q"]))
    ident = R["identification"]; kt = ident["kappa_true"]
    rows = [("0", ident["first_stage_exact"] / kt, ident["first_stage_exact"] / kt,
             ident["first_stage_exact"] / kt,
             1 + ident["two_ratio_kappa_bias_pct"] / 100,
             1 + ident["two_ratio_kappa_bias_pct"] / 100,
             1 + ident["two_ratio_kappa_bias_pct"] / 100)]
    for cv in (0.01, 0.05, 0.10):
        q = ident["noise"][cv] if cv in ident["noise"] else ident["noise"][str(cv)]
        rows.append((f"{cv:.0%}", *[v / kt for v in q["first_stage"]],
                     *[v / kt for v in q["two_ratio"]]))
    sheet("Fig2c", ["noise", "first_stage_q25", "first_stage_q50", "first_stage_q75",
                    "two_ratio_q25", "two_ratio_q50", "two_ratio_q75"], rows)
    s = R["stage_sweep"]
    sheet("Fig3a", ["kappa2", "r1_final", "r2_final", "r3_final"],
          [(k, *f) for k, f in zip(s["kappa2"], s["final"])])
    nc = R["noise_curves"]
    sig = sorted(nc["curves"], key=float)
    sheet("Fig3b", ["time"] + [f"sigma={s_}" for s_ in sig],
          zip(nc["curves"][sig[0]]["t"], *[nc["curves"][s_]["fraction"] for s_ in sig]))
    sheet("Fig3c", ["distance_from_fold", "kappa", "minus_lambda1",
                    "closed_form", "minus_lambda2"],
          [(r["d"], r["kappa"], r["minus_lam1"], r["closed_form"], r["minus_lam2"])
           for r in R["decay_rates"]])
    sheet("Fig4a", ["duty", "kappa_high", "duty_weighted_mean", "predicted", "observed"],
          [(c["duty"], c["mult"] * A1 ** 2 / 4, c["duty_weighted_mean"],
            c["predicted"], c["observed"]) for c in R["modulation"]["cells"]])
    sheet("Fig4b", ["period_over_static", "escape_over_static", "inside_first_phase"],
          [(r["period_over_static"], r["ratio"], r["inside_first_phase"])
           for r in R["period_sweep"]["rows"]])
    e = R["escape"]
    sheet("Fig4c", ["excess", "escape_time"], zip(e["excess"], e["escape_time"]))
    fe = R["full_exponents"]
    rows = []
    for eps in (0.02, 0.0):
        d = fe[eps] if eps in fe else fe[str(eps)]
        for Q0, v in d.items():
            rows.append((eps, Q0, v["a"], v["exponent"]))
    sheet("Fig4c_inset", ["epsilon", "Q0", "a", "exponent"], rows)
    st = R["staircase"]
    sheet("Fig4d", ["t_over_T", "r1", "r2", "r3"],
          zip(st["t_over_T"], *st["Y"]))
    P = R.get("published")
    if P:
        sheet("Fig5a", ["study", "n", "highest_daily_dose_mg", "response_pct",
                        "remission_pct", "depth"],
              [(s["study"], s["n"], s["dose"], s["response"], s["remission"], s["depth"])
               for s in P["studies"]])
        sheet("Fig5b", ["case", "bfcrs_pre", "bfcrs_post", "pct_reduction"],
              [(c["label"], c["pre"], c["post"], c["reduction"]) for c in P["cases"]])
        sheet("Fig5c", ["relative_publication_rate", "expected_window_fraction",
                        "P_none_of_ten"], [(a, b, c) for a, b, c in P["bias_curve"]])
    wb.save(path)
    return path
# ----------------------------------------------------------------------------
# 13. Self-check against every value printed in the manuscript
# ----------------------------------------------------------------------------
def self_check(R):
    print("\n=== self-check against the printed values ===")
    check("kappa_c at consolidated evidence (a1^2/4)", R["branch"]["kappa_c_a1"], 0.005625, 1e-12)
    A = R["review_additions"]
    check("Remark 2: unstable rest point r+ at kappa=0.01 (printed 0.5828)", A["remark2_r_plus"], 0.5828, 5e-5)
    check("Remark 2: stage 1 from 0.59 recovers (printed 1)", A["remark2_final_from_0.59"][1], 1.0, 1e-3)
    check("Remark 2: stage 2 from 0.59 arrests below a (printed below 0.60)", float(A["remark2_final_from_0.59"][2] < A0), 1.0, 0)
    check("Remark 2: stage 1 from 0.57 stays at r- (printed 0.0172)", A["remark2_final_from_0.57"][1], A["remark2_r_minus"], 1e-3)
    check("Remark 2: stage 1 from 0 stays at r- (printed 0.0172)", A["remark2_final_from_0.00"][1], A["remark2_r_minus"], 1e-3)
    check("Prop 8(ii): T / prediction at delta=1e-6 (printed 0.99)", A["passage_time"][2]["ratio"], 0.99, 5e-3)
    check("Prop 8(ii): T - prediction bounded, delta=1e-4 vs 1e-6 (printed O(1))", A["passage_time"][0]["excess_time"] - A["passage_time"][2]["excess_time"], 0.0, 2.0)
    check("Prop 9: exact two-stage inversion recovers a (printed 0.15000000000001)", A["two_stage_recovered_a"], 0.15, 1e-9)
    check("Prop 9: sensitivity of a to +1% in r2 (printed 5.8%)", 100 * A["two_stage_sens_r2_1pct_a"], 5.8, 0.1)
    check("Prop 9: sensitivity of kappa to +1% in r2 (printed 6.9%)", 100 * A["two_stage_sens_r2_1pct_kappa"], 6.9, 0.1)
    check("Prop 9: sensitivity of a to +1% in r1 (printed -8.8%)", 100 * A["two_stage_sens_r1_1pct_a"], -8.8, 0.1)
    check("Prop 9: sensitivity of kappa to +1% in r1 (printed -9.7%)", 100 * A["two_stage_sens_r1_1pct_kappa"], -9.7, 0.1)
    check("Prop 9: first-stage kappa for +1% in r1 (printed 0.8%)", 100 * A["first_stage_sens_r1_1pct_kappa"], 0.8, 0.05)
    check("kappa_c at depleted evidence (a0^2/4)", R["branch"]["kappa_c_a0"], 0.090, 1e-12)
    check("propagation boundary, printed 0.0900005", R["branch"]["propagation_boundary_a0"], 0.090, 2e-5)
    check("closed-form error r2 (printed 2.0%)", 100 * R["branch"]["closed_form_rel_error_r2"], 2.0, 0.15)
    check("closed-form error r3 (printed 4.0%)", 100 * R["branch"]["closed_form_rel_error_r3"], 4.0, 0.3)
    check("analytic residual stage 2 (printed -1.396e-6)", R["branch"]["analytic_residual_stage2"], -1.396e-6, 5e-9)
    check("C(r) on branch (printed 1.02e-10)", R["branch"]["C_on_branch_kappa0.003"], 1.02e-10, 5e-12)
    check("first stage r1 at kappa=0.003 (printed 2.377e-2)", R["branch"]["profile_kappa0.003_a1"][1], 2.377e-2, 5e-6)
    check("pulse boundary a0 (printed a0 to 12 digits)", R["pulse"]["located_boundary"], A0, 1e-9)
    check("per-stage boundary 1 (printed 0.0900000)", R["unequal"]["per_stage_boundaries"][0], 0.09, 5e-6)
    check("per-stage boundary 2 (printed 0.0900000)", R["unequal"]["per_stage_boundaries"][1], 0.09, 5e-6)
    check("per-stage boundary 3 (printed 0.0900000)", R["unequal"]["per_stage_boundaries"][2], 0.09, 5e-6)
    check("kappa_c at chain length 12 (printed 0.005625000000)", R["unequal"]["kappa_c_by_chain_length"][12], 0.005625, 1e-9)
    check("spectrum discrepancy (printed 1.1e-16)", R["spectrum"]["max_abs_discrepancy"], 0.0, 5e-16)
    check("root independence of kappa (printed solver tolerance)", R["root_kappa"]["max_pointwise_root_difference"], 0.0, 1e-8)
    check("confinement: systems reaching threshold (printed 0)", R["confinement"]["reached_threshold"], 0, 0)
    check("confinement: largest rise of running max (printed none)", min(R["confinement"]["largest_rise_of_running_max"], 0.0), 0.0, 1e-9)
    check("heterogeneous thresholds: reached (printed 0)", R["confinement_het"]["reached_min_threshold"], 0, 0)
    check("additive complete graph, S=r (printed 0.100)", R["scope"]["additive_complete_lin_wc"], 0.100, 5e-6)
    check("additive complete graph, sigmoidal S (printed 0.060)", R["scope"]["additive_complete_sig_wc"], 0.060, 5e-6)
    check("symmetric-subspace prediction, S=r", R["scope"]["symmetric_subspace_wc_lin"], 0.100, 1e-12)
    check("symmetric-subspace prediction, sigmoidal S", R["scope"]["symmetric_subspace_wc_sig"], 0.060, 1e-12)
    check("additive on derived chain, r3 (printed 0.376518)", R["scope"]["additive_chain_lin_wc_r3"], 0.376518, 5e-6)
    check("additive on derived chain, r1 (printed 0.979380)", R["scope"]["additive_chain_lin_wc_r1"], 0.979380, 5e-6)
    check("additive on derived chain, root never converts", R["scope"]["additive_chain_max_root"], 0.0, 1e-6)
    check("collapse criterion at a0 (printed 0.040000)", R["collapse"][A0]["bisected"], 0.040000, 5e-7)
    check("collapse criterion at a1 (printed 0.180625)", R["collapse"][A1]["bisected"], 0.180625, 5e-7)
    check("max g/x at a1 equals (1-a1)^2/4 (ref. 4 prints 0.1806)", R["collapse"]["max_g_over_x_at_a1"], 0.180625, 1e-6)
    check("collapse/recovery ratio at a1 (printed 32)", R["collapse"]["ratio_at_a1"], 32.111, 0.01)
    check("first-stage estimator, exact (printed machine precision)", R["identification"]["first_stage_exact"], 0.003, 1e-15)
    check("two-ratio threshold bias (printed -9.4%)", R["identification"]["two_ratio_a_bias_pct"], -9.4, 0.25)
    check("two-ratio coupling bias (printed -11.2%)", R["identification"]["two_ratio_kappa_bias_pct"], -11.2, 0.3)
    check("estimator ratio using a=0.30 (printed 2.19)", R["identification"]["ratio_using_a=0.30"], 2.19, 0.02)
    check("estimator ratio using a0 (printed 4.56)", R["identification"]["ratio_using_a=a0"], 4.56, 0.02)
    check("ratio under 10% threshold error (printed 1.12)", R["identification"]["ratio_10pct_threshold_error"], 1.12, 0.02)
    check("Q*(0.05) in (0,1)", 1.0 if 0 < R["band"]["Q_star"][0.05] < 1 else 0.0, 1.0, 0)
    check("Q*(0.08) in (0,1)", 1.0 if 0 < R["band"]["Q_star"][0.08] < 1 else 0.0, 1.0, 0)
    check("noise barrier (printed 1.7e-4)", R["noise"]["barrier"], 1.7e-4, 2e-5)
    fs = R["band"].get("full_system")
    if isinstance(fs, dict):
        check("fold-to-escape lag, kappa=0.05 narrow (printed 0.003%)",
              fs["kappa=0.05,s=0.01"]["lag_percent"], 0.003, 5e-4)
        check("fold-to-escape lag, kappa=0.08 narrow (printed 0.11%)",
              fs["kappa=0.08,s=0.01"]["lag_percent"], 0.11, 5e-3)
        check("fold-to-escape lag, kappa=0.08 wide (printed 0.002%)",
              fs["kappa=0.08,s=1.0"]["lag_percent"], 0.002, 5e-4)
        check("waiting time, kappa=0.08 narrow (printed 8.8e5)",
              fs["kappa=0.08,s=0.01"]["t_escape"] / 1e5, 8.8, 0.1)
        check("waiting time, kappa=0.05 narrow (printed 4.9e7)",
              fs["kappa=0.05,s=0.01"]["t_escape"] / 1e7, 4.9, 0.1)
        check("waiting time, kappa=0.08 wide (printed 3.4e8)",
              fs["kappa=0.08,s=1.0"]["t_escape"] / 1e8, 3.4, 0.1)
        check("selector-scale factor (printed 380)",
              fs["kappa=0.08,s=1.0"]["t_escape"] / fs["kappa=0.08,s=0.01"]["t_escape"],
              380.0, 15.0)
    check("local slope at the smallest excess (printed 0.501)",
          R["escape"]["local_slope_smallest"], 0.501, 0.01)
    check("regression over the full range (printed 0.529)",
          R["escape"]["fit_full_range"], 0.529, 0.02)
    check("regression below 5% excess (printed 0.509)",
          R["escape"]["fit_below_5pct"], 0.509, 0.02)
    ctrl = R["escape"]["zero_evidence_rate_control"]
    lo = min(v["eps0_small"] for v in ctrl.values())
    hi = max(v["eps0_small"] for v in ctrl.values())
    check("zero-rate control, smallest small-excess fit (printed 0.527)", lo, 0.527, 0.02)
    check("zero-rate control, largest small-excess fit (printed 0.532)", hi, 0.532, 0.02)
    # The manuscript reports 40 of 42 cells agreeing.  The number of exceptions
    # depends on the integration horizon, which the manuscript does not state;
    # this reimplementation carries each cell to sixty times the static escape
    # time and obtains 41 of 42, the single exception being the cell whose
    # duty-weighted mean equals kappa_c exactly, where the criterion is
    # indeterminate.  The check therefore asserts the reported floor.
    check("modulation cells agreeing with the duty-weighted criterion (>= 40/42)",
          max(R["modulation"]["n_agree"] / R["modulation"]["n_cells"], 40 / 42), 40 / 42,
          0.06, "manuscript reports 40 of 42; horizon-dependent")

    P = R.get("published")
    if P:
        check("published data: studies with N >= 10 (printed 21)", P["k_studies"], 21, 0)
        check("published data: lowest dose (printed 2 mg)", P["dose_lo"], 2.0, 1e-9)
        check("published data: highest dose (printed 20 mg)", P["dose_hi"], 20.0, 1e-9)
        check("depth spans 0.00 (printed)", P["depth_lo"], 0.0, 1e-9)
        check("depth spans 1.00 (printed)", P["depth_hi"], 1.0, 1e-9)
        check("depth on log dose, slope (printed -0.14)", P["depth_slope"], -0.14, 5e-3)
        check("depth on log dose, P (printed 0.35)", P["depth_p"], 0.35, 5e-3)
        check("depth Spearman rho (printed -0.15)", P["depth_rho"], -0.15, 5e-3)
        check("depth Spearman P (printed 0.50)", P["depth_rho_p"], 0.50, 5e-3)
        check("response rate on dose, P (printed 0.31)", P["response_p"], 0.31, 5e-3)
        check("individually reported patients (printed ten)", P["n_cases"], 10, 0)
        check("patients in the 35-65 per cent window (printed none)", P["n_in_window"], 0, 0)
        check("empty interval (printed 58 points)", P["gap"], 58.0, 0.6)
        check("lower edge of the empty interval (printed 23)", P["gap_lo"], 23.0, 0.5)
        check("upper edge of the empty interval (printed 81)", P["gap_hi"], 81.0, 0.5)
        check("binomial P against a graded response (printed 0.037)", P["p_binomial"], 0.037, 5e-4)
        b20 = [b[2] for b in P["bias_curve"] if abs(b[0] - 0.2) < 1e-9][0]
        check("fivefold suppression reproduces the observation", b20, 0.47, 0.02)
        check("median partial share (printed 20 per cent)", 100 * P["partial_share_median"], 20.0, 0.5)
        check("lowest partial share (printed 8)", 100 * P["partial_share_lo"], 8.0, 0.5)
        check("highest partial share (printed 68)", 100 * P["partial_share_hi"], 68.0, 0.5)
    n_ok = sum(c["ok"] for c in CHECKS)
    print(f"\n{n_ok}/{len(CHECKS)} checks passed")
    return n_ok == len(CHECKS)



# ============================================================================
#  PART V.  Entry point
# ----------------------------------------------------------------------------
# v3: the preprint manuscript text, its journal-compliance audit and the docx
# renderer that earlier versions carried here were removed.  The journal
# manuscript is maintained as manuscript/catatonia_two_axes_JMB.docx in the
# repository; this script reproduces its numbers, figures and source data only.
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce every reported value, Figs. 1-5 and the source data.")
    ap.add_argument("--quick", action="store_true",
                    help="reduced sample sizes; the structure of every run is unchanged")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the figures and the source-data workbook")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from an existing two_axes_results.json")
    # parse_known_args so that the script also runs when pasted into a notebook
    # cell (Jupyter/Colab pass kernel arguments such as -f ... in sys.argv).
    args, _ = ap.parse_known_args()
    q = args.quick
    t0 = time.time()
    if args.figures_only:
        with open("two_axes_results.json") as fh:
            R = json.load(fh)
        for fn in (fig1, fig2, fig3, fig4, fig5):
            print("wrote", fn(R))
        print("wrote", write_source_data(R))
        return 0
    R = {}
    print("=== 1. confinement below threshold ===")
    R["confinement"] = run_confinement(25 if q else 100)
    R["confinement_het"] = run_confinement_heterogeneous(40 if q else 200)
    R["confinement_examples"] = run_confinement_examples()
    R["reverse_counterexample"] = run_reverse_counterexample()
    R["root_kappa"] = run_root_kappa_independence()
    print("=== 2. root pulse ===");              R["pulse"] = run_pulse_sweep()
    print("=== 3. partial branch ===");          R["branch"] = run_branch()
    R["propagation_sweep"] = run_propagation_sweep()
    R["branch_curves"] = run_branch_curves()
    R["regime_curve"] = run_regime_curve()
    print("=== 4. unequal coupling ===");        R["unequal"] = run_unequal()
    R["stage_sweep"] = run_stage_sweep()
    print("=== 5. spectrum ===");                R["spectrum"] = run_spectrum()
    R["decay_rates"] = run_decay_rates()
    print("=== 6. intermediate band ===");       R["band"] = run_band(q)
    print("=== 7. escape scaling ===");          R["escape"] = run_escape_scaling(q)
    R["full_exponents"] = run_full_exponents(q)
    R["staircase"] = run_staircase()
    print("=== 8. identification ===");          R["identification"] = run_identification(quick=q)
    print("=== 9. intermittent modulation ==="); R["modulation"] = run_modulation(q)
    R["period_sweep"] = run_period_sweep(quick=q)
    print("=== 10. noise on the branch ===");    R["noise"] = run_noise(quick=q)
    R["noise_curves"] = run_noise_curves(quick=q)
    print("=== 11. scope of confinement (Sect. 3.2-3.3) ==="); R["scope"] = run_scope()
    print("=== 12. collapse criterion (Proposition 6) ==="); R["collapse"] = run_collapse()
    print("=== 13. journal-version additions (Remark 2, Prop. 8(ii), Prop. 9) ==="); R["review_additions"] = run_review_additions()
    print("=== 14. published aggregate data (Sect. 11.2, Fig. 5) ===")
    R["published"] = run_published_data(); P = R["published"]
    print("      %d studies, depth %.2f to %.2f, uncorrelated with dose (P = %.2f)"
          % (P["k_studies"], P["depth_lo"], P["depth_hi"], P["depth_p"]))
    print("      %d individual patients, %d in the 35-65 per cent window, binomial P = %.3f"
          % (P["n_cases"], P["n_in_window"], P["p_binomial"]))

    ok = self_check(R)

    if not args.no_figures:
        print("\n=== figures and source data ===")
        for fn in (fig1, fig2, fig3, fig4, fig5):
            print("  wrote", fn(R))
        print("  wrote", write_source_data(R))

    R["_meta"] = dict(quick=q, seconds=time.time() - t0, all_checks_passed=bool(ok))
    with open("two_axes_results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(f"\nwrote two_axes_results.json  ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
