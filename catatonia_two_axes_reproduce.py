#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction script for

    "Catatonia recovery separates two intervention axes that constrain
     electroconvulsive therapy"
    H. Saito

Self-contained: runs every numerical analysis reported in the manuscript and
regenerates Figs. 1-4.

    python saito_two_axes_reproduce.py            # everything (~15 min)
    python saito_two_axes_reproduce.py --quick    # coarser grids (~3 min)
    python saito_two_axes_reproduce.py --figures  # figures only, from results.json
    python saito_two_axes_reproduce.py --verify   # check every value the paper reports
    python saito_two_axes_reproduce.py --docx T.docx   # write the manuscript and the SI
    python saito_two_axes_reproduce.py --audit    # Nature Communications compliance

The flags compose: whatever a later step needs is produced first if it is
missing, so --quick --figures --verify --audit --docx T.docx is a full run.

This one file contains everything: the model, every analysis, the figures, the
value-by-value verification, and the full text of the manuscript and of the
Supplementary Information together with the code that renders them to Word.
--docx needs a template .docx only for its page setup, styles and footers; none
of the template's text is used.

Step 5 (intermittent modulation) is the slowest, roughly half the total, and
prints per-cell progress so a long run is not mistaken for a hang.

In a notebook (Jupyter, Colab) the command line is ignored, because the kernel
appends its own arguments.  Pasting the file into a cell runs the full analysis;
to choose a mode call the helper instead:

    run(quick=True)      run()      run(verify=True)      run(figures=True)

Outputs (written to --outdir, default the working directory)
    results.json   every reported number, including the Fig. 4d trajectory
    staircase.npy  the same trajectory at full resolution (skipped if unwritable)
    Fig1/2/3/4 .pdf and .png
    Source_Data_Figs1-4.xlsx   one sheet per figure panel, as plotted

Model (ref. 4)

    dr0/dt = r0(1-r0)(r0-a(Q)) + u(t)
    drk/dt = rk(1-rk)(rk-a(Q)) + kappa*(r_{k-1} - rk),   k = 1,2,3
    dQ/dt  = eps*[ chi+(z) C(r) (1-Q) - rho*chi-(z) Q ],  z = r0 - a(Q)
    a(Q)   = a0 - (a0-a1) Q^p,      C(r) = prod_j r_j

Parameters are the illustrative set published in ref. 4, used unmodified:
a0 = 0.60, a1 = 0.15, p = 2, eps = 0.02, rho = 1, and the one-sided
regularisation chi_pm(z) = exp(-s/z^2) on its own half-line, at the two
published scales s = 0.01 and s = 1.  No parameter was introduced here.

WHAT IS EXACT AND WHAT IS NOT
-----------------------------
* The confinement result is a maximum principle and needs no chain structure:
  for dr_i/dt = r_i(1-r_i)(r_i-a) + sum_j w_ij (r_j - r_i) with any w_ij >= 0 on
  any graph, the coupling term is non-positive at an index attaining the maximum,
  so D+ max_k r_k <= M(1-M)(M-a) < 0 while 0 < M < a.  {max_k r_k < a} is forward
  invariant for every topology, every weight, and both directions of coupling.
  `analysis_max_principle` checks this on four graph families and also gives the
  counterexample that delimits it: with the root below threshold but downstream
  capacities above it, reverse coupling alone carries the root across.
* The first-stage equilibrium is exact.  With r0 = 1,
      r1(1-r1)(r1-a) + kappa(1-r1) = (1-r1)[r1^2 - a r1 + kappa] = 0,
  giving r1 = [a - sqrt(a^2-4 kappa)]/2 and kappa_c = a^2/4 exactly.
* Stages 2 and 3 are NOT exact in closed form.  Substituting
  rk = [a - sqrt(a^2 - 4 kappa r_{k-1})]/2 leaves the analytic residual
      f_k = kappa * rk * (r_{k-1} - 1),
  which is non-zero because (1 - rk) is not a common factor once r_{k-1} < 1.
  At kappa = 0.003 the closed form overstates r2 by 2.0% and r3 by 4.0%.
  `true_branch` therefore solves the equilibrium numerically and every reported
  branch quantity uses it; `closed_branch` is kept only to quantify that gap.
* Consequently the two-stage-ratio inversion is biased on the true branch
  (about -9% in a and -11% in kappa at kappa = 0.003) even at zero noise: it was
  derived from the approximate branch.  The primary estimator, a from the
  tolerated excursion and kappa = a*r1 - r1^2 from the first stage alone, is
  exact on the true branch to machine precision, because stage one factorises.
* First-order averaging over the rectangular modulation is exact in kappa,
  because the field is linear in kappa.  The duty-weighted criterion is therefore
  expected, and the informative content is where it fails, not where it holds.

Requires numpy, scipy, matplotlib.  Deterministic (seed 20260810).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Parameters of ref. 4
# --------------------------------------------------------------------------
A0, A1 = 0.60, 0.15          # threshold at no evidence / at full evidence
P_EXP = 2.0                  # threshold interpolation exponent
EPS = 0.02                   # evidence rate
RHO = 1.0                    # attenuation rate
KAPPA_REF = 0.6              # illustrative coupling
KAPPA_UNTREATED = 0.003      # coupling of the illustrative partial branch
SEED = 20260810

KC = A1 ** 2 / 4             # critical coupling at consolidated evidence
KC_DEPLETED = A0 ** 2 / 4    # critical coupling at depleted evidence
K_STAR = max((1 - a + a * a) / 3 for a in np.linspace(A1, A0, 10001))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def a_of(Q):
    """Evidence-dependent threshold."""
    return A0 - (A0 - A1) * Q ** P_EXP


def capacity_field(t, r, a, kappa, u=0.0):
    """Feed-forward capacity subsystem with the threshold held fixed."""
    d = r * (1 - r) * (r - a)
    d[0] += u
    d[1:] += kappa * (r[:-1] - r[1:])
    return d


def graph_field(t, r, a, W):
    """Same bistable law on an arbitrary graph: W[i, j] >= 0 is the weight j -> i.

    `a` may be a scalar or one threshold per node; the maximum principle holds
    in both cases, with the invariant set {max r < a} weakening to {max r < min a}.
    """
    return r * (1 - r) * (r - a) + (W * (r[None, :] - r[:, None])).sum(1)


def chi(z, s):
    """One-sided regularisation exp(-s/z^2); magnitude only, sign handled by caller."""
    az = abs(z)
    return float(np.exp(-s / az ** 2)) if az > 1e-12 else 0.0


def full_field_vec_factory(kap, s):
    """Five-dimensional field with a coupling per stage."""
    kap = np.asarray(kap, float)

    def f(t, y):
        r = y[:4]
        Q = min(max(y[4], 0.0), 1.0)
        a = a_of(Q)
        z = r[0] - a
        d = np.empty(5)
        d[:4] = r * (1 - r) * (r - a)
        d[1:4] += kap * (r[:3] - r[1:4])
        cp = chi(z, s) if z > 0 else 0.0
        cm = chi(z, s) if z < 0 else 0.0
        d[4] = EPS * (cp * float(np.prod(r)) * (1 - Q) - RHO * cm * Q)
        return d
    return f


def full_field_factory(kappa, s, eps=None):
    """Full five-dimensional field, evidence channel active.

    eps = 0 switches the evidence channel off and changes nothing else.  That
    is the control used below to separate pre-asymptotic curvature of the fold
    from the effect of evidence feedback on the escape-time exponent.
    """
    rate = EPS if eps is None else float(eps)

    def f(t, y):
        r = y[:4]
        Q = min(max(y[4], 0.0), 1.0)
        a = a_of(Q)
        z = r[0] - a
        d = np.empty(5)
        d[:4] = r * (1 - r) * (r - a)
        d[1:4] += kappa * (r[:3] - r[1:4])
        cp = chi(z, s) if z > 0 else 0.0
        cm = chi(z, s) if z < 0 else 0.0
        d[4] = rate * (cp * float(np.prod(r)) * (1 - Q) - RHO * cm * Q)
        return d
    return f


def integrate(r0, a, kappa, tmax, u=0.0, rtol=1e-10, atol=1e-12, dense=False):
    return solve_ivp(capacity_field, (0, tmax), np.asarray(r0, float),
                     args=(a, kappa, u), method='LSODA',
                     rtol=rtol, atol=atol, dense_output=dense)


def first_stage(kappa, a=A1):
    """Exact low first-stage root; None when the fold has already occurred."""
    disc = a * a - 4 * kappa
    if disc < 0:
        return None
    return (a - np.sqrt(disc)) / 2


def closed_branch(kappa, a=A1):
    """Stage-wise closed form.  EXACT AT STAGE 1 ONLY (see module docstring)."""
    r1 = first_stage(kappa, a)
    if r1 is None:
        return None
    r2 = (a - np.sqrt(a * a - 4 * kappa * r1)) / 2
    r3 = (a - np.sqrt(a * a - 4 * kappa * r2)) / 2
    return np.array([1.0, r1, r2, r3])


def branch_residual(r, a, kappa):
    """Vector field on the downstream stages; equals kappa*r_k*(r_{k-1}-1) for closed_branch."""
    return capacity_field(0.0, np.asarray(r, float).copy(), a, kappa)[1:]


def true_branch(kappa, a=A1):
    """Exact partial equilibrium with the root held at 1.

    Stage 1 is the quadratic above and is exact.  Each later stage satisfies the
    CUBIC  r_k(1-r_k)(r_k-a) + kappa(r_{k-1} - r_k) = 0, not the quadratic the
    stage-wise closed form assumes.  With 0 < r_{k-1} < a the cubic changes sign
    on (0, a), so the low root is bracketed and brentq returns it exactly.
    """
    r1 = first_stage(kappa, a)
    if r1 is None:
        return None
    r = [1.0, float(r1)]
    for _ in (2, 3):
        prev = r[-1]

        def h(x, prev=prev):
            return x * (1 - x) * (x - a) + kappa * (prev - x)
        if h(0.0) <= 0 or h(a) >= 0:
            return None
        r.append(float(brentq(h, 0.0, a, xtol=1e-16, rtol=8.9e-16, maxiter=200)))
    return np.array(r)


def identify_two_ratio(r1, r2):
    """Inversion from two consecutive stage ratios.  Biased on the true branch."""
    a = (r1 ** 3 - r2 ** 2) / (r1 ** 2 - r2)
    return a, a * r1 - r1 ** 2


def identify_primary(r1, a):
    """kappa from the first stage alone, given the threshold.  Exact on the true branch."""
    return a * r1 - r1 ** 2


def annihilation_Q(kappa):
    """Evidence level at which the partial branch is annihilated for fixed kappa.

    The branch exists while kappa <= a(Q)^2/4, i.e. while a(Q) >= 2*sqrt(kappa).
    Setting a(Q*) = 2*sqrt(kappa) and inverting a(Q) gives
        Q* = [ (a0 - 2 sqrt(kappa)) / (a0 - a1) ]^(1/p),
    which tends to 1 as kappa -> (a1^2/4)+ and to 0 as kappa -> (a0^2/4)-.
    Returns None below the band, where the branch survives at every attainable Q.
    """
    if kappa <= KC:
        return None
    if kappa >= KC_DEPLETED:
        return 0.0
    return float(((A0 - 2 * np.sqrt(kappa)) / (A0 - A1)) ** (1.0 / P_EXP))


# --------------------------------------------------------------------------
# 1. Confinement below threshold: a maximum principle, valid on any graph
# --------------------------------------------------------------------------
def _random_weights(rng, n, family):
    W = rng.random((n, n)) * rng.choice([0.0, 0.5, 3.0, 30.0], size=(n, n),
                                        p=[0.35, 0.25, 0.25, 0.15])
    np.fill_diagonal(W, 0.0)
    if family == 'feedforward chain':
        W = np.diag(np.diag(W, -1), -1)      # same mixture, sub-diagonal only
    elif family == 'symmetric':
        W = W + W.T
    elif family == 'reverse only':
        W = np.tril(W, -1).T * 3
    return W


def analysis_max_principle(rng, quick=False):
    families = ['feedforward chain', 'symmetric', 'reverse only', 'general directed']
    ntrial = 25 if quick else 100
    rows = []
    for family in families:
        viol, rise, nodes = 0, 0.0, []
        for _ in range(ntrial):
            n = int(rng.integers(3, 7))
            W = _random_weights(rng, n, family)
            r0 = rng.random(n) * A0 * 0.999          # every capacity below threshold
            y = solve_ivp(graph_field, (0, 600), r0, args=(A0, W), method='LSODA',
                          rtol=1e-9, atol=1e-12).y
            if y.max() >= A0:
                viol += 1
            rise = max(rise, float(y.max() - r0.max()))
            nodes.append(n)
        rows.append(dict(family=family, trials=ntrial, crossings=viol,
                         max_rise_of_maximum=rise,
                         nodes_min=int(min(nodes)), nodes_max=int(max(nodes))))
    # The counterexample that delimits the statement: the maximum is NOT below a.
    counter = []
    for kf, kb in ((0.01, 0.0), (0.01, 0.02), (0.01, 0.10), (0.10, 0.30), (0.0, 0.10)):
        def bi(t, r, a):
            d = r * (1 - r) * (r - a)
            d[1:] += kf * (r[:-1] - r[1:])
            d[:-1] += kb * (r[1:] - r[:-1])
            return d
        f = solve_ivp(bi, (0, 4000), np.array([0.30, 0.90, 0.90, 0.90]), args=(A0,),
                      method='LSODA', rtol=1e-10, atol=1e-13).y[:, -1]
        counter.append(dict(kappa_forward=kf, kappa_reverse=kb,
                            final=[float(x) for x in f], root_crossed=bool(f[0] > 0.5)))
    # Feed-forward special case: the root equation contains no kappa at all.
    dev = 0.0
    for _ in range(10 if quick else 30):
        r0 = rng.random(4) * A0 * 0.999
        ts = np.linspace(0, 300, 601)
        base = integrate(r0, A0, 0.6, 300.0, dense=True).sol(ts)[0]
        for kappa in (3.0, 30.0):
            alt = integrate(r0, A0, kappa, 300.0, dense=True).sol(ts)[0]
            dev = max(dev, float(np.max(np.abs(base - alt))))
    # Thresholds need not be equal.  At an index attaining the maximum the
    # coupling term is still non-positive, so D+M <= M(1-M)(M - a_that_index),
    # which is negative as long as M is below EVERY threshold: the invariant set
    # weakens from {max r < a} to {max r < min a} and nothing else changes.
    het_viol, het_rise, ntrial_h = 0, 0.0, (50 if quick else 200)
    for _ in range(ntrial_h):
        n = int(rng.integers(3, 7))
        W = _random_weights(rng, n, 'general directed')
        aa = rng.uniform(A1, A0, n)
        r0 = rng.random(n) * aa.min() * 0.999
        y = solve_ivp(graph_field, (0, 600), r0, args=(aa, W), method='LSODA',
                      rtol=1e-9, atol=1e-12).y
        if (y.max(0) >= aa.min()).any():
            het_viol += 1
        het_rise = max(het_rise, float(y.max() - r0.max()))

    return dict(max_principle=rows,
                max_principle_total_crossings=sum(r['crossings'] for r in rows),
                max_principle_total_trials=sum(r['trials'] for r in rows),
                heterogeneous_thresholds=dict(trials=ntrial_h, crossings=het_viol,
                                              max_rise_of_maximum=het_rise,
                                              a_min=float(A1), a_max=float(A0)),
                reverse_coupling_counterexample=counter,
                root_trajectory_deviation=dev)


# --------------------------------------------------------------------------
# 2. The root excursion: a genuine pulse, and the crossing criterion
# --------------------------------------------------------------------------
def _pulse_outcome(amp, tau, kappa=KAPPA_REF, a=A0, tmax=4000.0):
    """Rectangular u(t) = amp on [0, tau] from the collapsed chain."""
    def f(t, r):
        return capacity_field(t, r, a, kappa, amp if t < tau else 0.0)
    s = solve_ivp(f, (0, tmax), np.zeros(4), method='LSODA', rtol=1e-10, atol=1e-13,
                  dense_output=True)
    return float(s.sol(tau)[0]), s.y[:, -1]


def analysis_pulse(quick=False):
    taus = [0.5, 2.0] if quick else [0.25, 0.5, 1.0, 2.0, 5.0]
    rows, thresholds = [], {}
    for tau in taus:
        lo, hi = 1e-4, 5.0
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            if _pulse_outcome(mid, tau)[1][3] > 0.5:
                hi = mid
            else:
                lo = mid
        amp_c = 0.5 * (lo + hi)
        r0_end, _ = _pulse_outcome(amp_c, tau)
        thresholds[str(tau)] = dict(critical_amplitude=amp_c, root_at_pulse_end=r0_end)
        for amp in (0.8 * amp_c, 0.99 * amp_c, 1.01 * amp_c, 1.2 * amp_c):
            end, fin = _pulse_outcome(amp, tau)
            rows.append(dict(tau=tau, amplitude=amp, root_at_pulse_end=end,
                             recovered=bool(fin[3] > 0.5), final=[float(x) for x in fin]))
    # every (amplitude, duration) pair converts iff the pulse leaves the root above a0
    consistent = all(r['recovered'] == (r['root_at_pulse_end'] > A0) for r in rows)
    # initial-condition sweep, for the figure; the invariant surface itself is excluded
    step = 0.02 if quick else 0.01
    grid = [round(v, 4) for v in np.arange(0.40, 0.8001, step) if abs(v - A0) > 1e-9]
    grid += [round(A0 - d, 6) for d in (0.02, 0.01, 0.004, 0.001)]
    grid += [round(A0 + d, 6) for d in (0.001, 0.004, 0.01, 0.02)]
    grid = sorted(set(grid))
    sweep = [dict(v=float(v),
                  final=[float(x) for x in integrate([v, 0, 0, 0], A0, KAPPA_REF, 4000.0).y[:, -1]])
             for v in grid]
    thr = brentq(lambda v: integrate([v, 0, 0, 0], A0, KAPPA_REF, 4000.0).y[3, -1] - 0.5,
                 0.55, 0.65, xtol=1e-12)
    thr_k = {}
    for kappa in (0.35, 0.6, 3.0):
        thr_k[str(kappa)] = brentq(
            lambda v: integrate([v, 0, 0, 0], A0, kappa, 4000.0).y[3, -1] - 0.5,
            0.55, 0.65, xtol=1e-12)
    return dict(pulse_thresholds=thresholds, pulse_rows=rows,
                pulse_criterion_consistent=bool(consistent),
                excursion_grid=sweep, excursion_threshold=float(thr),
                excursion_threshold_vs_kappa=thr_k)


# --------------------------------------------------------------------------
# 3. Partial branch, critical coupling, identification
# --------------------------------------------------------------------------
def _propagation_outcome(kappa, v=0.65, a=A0, tmax=2e4):
    return integrate([v, 0, 0, 0], a, kappa, tmax).y[3, -1]


def analysis_branch(rng, quick=False):
    comp = []
    for kappa in (0.001, 0.003, 0.005, 0.0056):
        c, t = closed_branch(kappa), true_branch(kappa)
        comp.append(dict(kappa=kappa,
                         closed=[float(x) for x in c], true=[float(x) for x in t],
                         rel_error=[0.0] + [float(c[i] / t[i] - 1) for i in (1, 2, 3)],
                         analytic_residual=[float(kappa * c[k] * (c[k - 1] - 1)) for k in (1, 2, 3)],
                         numeric_residual=[float(x) for x in branch_residual(c, A1, kappa)]))
    tb = true_branch(KAPPA_UNTREATED)
    back = integrate(tb + np.array([0, 1e-4, 1e-4, 1e-4]), A1, KAPPA_UNTREATED, 5000.0)

    prop = {str(k): [float(x) for x in integrate([0.65, 0, 0, 0], A0, k, 2e4).y[:, -1]]
            for k in (0.001, 0.003, 0.01, 0.05, 0.07, 0.085, 0.089, 0.0895,
                      0.0905, 0.091, 0.095, 0.11, 0.2, 0.291, 0.6)}
    prop_bd = brentq(lambda k: _propagation_outcome(k) - 0.5, 0.02, 0.20, xtol=1e-12)

    a_hat_c, k_hat_c = identify_two_ratio(*closed_branch(KAPPA_UNTREATED)[1:3])
    a_hat_t, k_hat_t = identify_two_ratio(*tb[1:3])
    bias = []
    for kappa in (0.001, 0.003, 0.005):
        t = true_branch(kappa)
        a_t, k_t = identify_two_ratio(t[1], t[2])
        bias.append(dict(kappa=kappa, a_hat=float(a_t), kappa_hat=float(k_t),
                         a_rel_error=float(a_t / A1 - 1), kappa_rel_error=float(k_t / kappa - 1),
                         primary_kappa_hat=float(identify_primary(t[1], A1)),
                         primary_rel_error=float(identify_primary(t[1], A1) / kappa - 1)))

    ndraw = 400 if quick else 2000
    noise = {}
    for pct in (0.01, 0.05, 0.10):
        sigma = np.sqrt(np.log1p(pct ** 2))          # lognormal with CV = pct, unit median-ish
        mu = -0.5 * sigma ** 2
        two, one = [], []
        for _ in range(ndraw):
            f1, f2 = np.exp(rng.normal(mu, sigma, 2))
            r1n, r2n = tb[1] * f1, tb[2] * f2
            try:
                a_h, k_h = identify_two_ratio(r1n, r2n)
            except ZeroDivisionError:
                a_h = k_h = np.nan
            if np.isfinite(a_h) and np.isfinite(k_h):
                two.append((a_h, k_h))
            one.append(identify_primary(r1n, A1))
        two = np.array(two)
        one = np.array(one)
        noise[str(pct)] = dict(
            two_ratio=dict(
                a_median=float(np.median(two[:, 0])),
                a_iqr=[float(np.percentile(two[:, 0], 25)), float(np.percentile(two[:, 0], 75))],
                k_median=float(np.median(two[:, 1])),
                k_iqr=[float(np.percentile(two[:, 1], 25)), float(np.percentile(two[:, 1], 75))]),
            primary=dict(
                k_median=float(np.median(one)),
                k_iqr=[float(np.percentile(one, 25)), float(np.percentile(one, 75))]))
    # The primary estimator kappa = a r1 - r1^2 needs the threshold PREVAILING on
    # the branch, a(Q).  An acute graded challenge reads the depleted-evidence
    # threshold a0 instead, which is a different quantity; the cost of that
    # substitution is a systematic factor, not a small bias.
    mis = []
    for a_used in (A1, 0.20, 0.30, 0.45, A0):
        k_hat = identify_primary(tb[1], a_used)
        mis.append(dict(a_used=float(a_used), kappa_hat=float(k_hat),
                        ratio=float(k_hat / KAPPA_UNTREATED)))
    # local sensitivity: 10% error in the prevailing threshold
    sens = identify_primary(tb[1], A1 * 1.10) / KAPPA_UNTREATED - 1.0

    return dict(kappa_c_closed_form=KC, kappa_c_depleted=KC_DEPLETED,
                K_star_ref4=float(K_STAR),
                threshold_misspecification=mis,
                threshold_sensitivity_10pct=float(sens),
                branch_comparison=comp,
                true_branch=[float(x) for x in tb],
                closed_branch=[float(x) for x in closed_branch(KAPPA_UNTREATED)],
                true_branch_residual=float(np.max(np.abs(branch_residual(tb, A1, KAPPA_UNTREATED)))),
                closed_branch_residual=float(np.max(np.abs(branch_residual(
                    closed_branch(KAPPA_UNTREATED), A1, KAPPA_UNTREATED)))),
                true_branch_return=float(np.max(np.abs(back.y[:, -1] - tb))),
                branch_exists={str(k): first_stage(k) is not None
                               for k in (0.005, 0.0056, 0.005625, 0.00563, 0.006)},
                propagation=prop, propagation_boundary=float(prop_bd),
                propagation_boundary_closed_form=KC_DEPLETED,
                identification_on_closed=dict(a_hat=float(a_hat_c), kappa_hat=float(k_hat_c)),
                identification_on_true=dict(a_hat=float(a_hat_t), kappa_hat=float(k_hat_t)),
                identification_bias=bias, identification_noise=noise)


# --------------------------------------------------------------------------
# 4. Escape time: scaling, transit times, staircase
# --------------------------------------------------------------------------
def escape_time(kappa, tmax=3e5, a=A1, start=None):
    y0 = (true_branch(KAPPA_UNTREATED, a) if start is None else np.asarray(start)).copy()

    def ev(t, r, *args):
        return r[3] - a
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(capacity_field, (0, tmax), y0, args=(a, kappa, 0.0),
                  method='LSODA', rtol=1e-11, atol=1e-14, events=ev)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else np.inf


def analysis_escape(quick=False):
    excess = np.array([1e-4, 1e-3, 1e-2, 1e-1, 8e-1] if quick else
                      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 8e-1])
    t = np.array([escape_time(KC * (1 + e)) for e in excess])
    slopes = [None] + [float(-(np.log(t[i]) - np.log(t[i - 1]))
                             / (np.log(excess[i]) - np.log(excess[i - 1])))
                       for i in range(1, len(excess))]
    sel = excess <= 0.05
    out = dict(escape_times=[dict(excess=float(e), t=float(x), local_slope=s)
                             for e, x, s in zip(excess, t, slopes)],
               exponent_full_range=float(-np.polyfit(np.log(excess), np.log(t), 1)[0]),
               exponent_small_excess=float(-np.polyfit(np.log(excess[sel]), np.log(t[sel]), 1)[0]),
               local_slope_at_smallest_excess=slopes[1])

    kap = KC * 1.05
    T = escape_time(kap)
    sol = solve_ivp(capacity_field, (0, T), true_branch(KAPPA_UNTREATED).copy(),
                    args=(A1, kap, 0.0), method='LSODA', rtol=1e-11, atol=1e-14,
                    dense_output=True)
    ts = np.linspace(0, T, 4001)
    ys = sol.sol(ts)
    try:
        np.save('staircase.npy', np.vstack([ts, ys]))
    except OSError as exc:
        print('      (staircase.npy not written: %s; trajectory is in results.json)' % exc)
    keep = np.linspace(0, len(ts) - 1, 401).astype(int)
    traj = [[float(ts[i])] + [float(ys[k][i]) for k in range(4)] for i in keep]

    cross, transit = {}, {}
    for k in (1, 2, 3):
        above = np.nonzero(ys[k] > A1)[0]
        cross['r%d' % k] = float(ts[above[0]] / T) if len(above) else 1.0
        hi = np.nonzero(ys[k] > 0.9)[0]
        transit['r%d' % k] = (float((ts[hi[0]] - ts[above[0]]) / T)
                              if len(above) and len(hi) else None)
    stasis = {}
    tot = ys[1] + ys[2] + ys[3]
    rate = np.gradient(tot, ts)
    for thr in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        stasis['%g' % thr] = float(np.mean(rate < thr * rate.max()))
    out['staircase'] = dict(T=float(T), crossings=cross, transit_a1_to_0p9=transit,
                            stasis_fraction_vs_threshold=stasis,
                            r3_at_escape=float(ys[3][-1]),
                            note='r3 crosses at 1.00 by construction: the escape time is '
                                 'defined as that crossing, and the chain is not recovered there',
                            trajectory=traj)
    return out


# --------------------------------------------------------------------------
# 5. Intermittent modulation; averaging is exact in kappa, so the failures matter
# --------------------------------------------------------------------------
def escape_time_modulated(kappa_hi, f, period, kappa_lo=KAPPA_UNTREATED, tmax=3e4, a=A1):
    start = true_branch(KAPPA_UNTREATED, a)

    def rhs(t, r):
        k = kappa_hi if (t % period) < f * period else kappa_lo
        return capacity_field(t, r, a, k)

    def ev(t, r):
        return r[3] - a
    ev.terminal, ev.direction = True, 1
    s = solve_ivp(rhs, (0, tmax), start.copy(), method='LSODA', rtol=1e-8, atol=1e-11,
                  events=ev, max_step=period / 12)
    return float(s.t_events[0][0]) if len(s.t_events[0]) else np.inf


def analysis_modulation(quick=False):
    t_ref = escape_time(KC * 1.2)
    period = t_ref / 50.0
    fs = [0.1, 0.3, 0.7] if quick else [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    khs = [0.008, 0.02, 0.08] if quick else [0.006, 0.008, 0.01, 0.02, 0.04, 0.08, 0.15]
    grid, total = [], len(fs) * len(khs)
    for f in fs:
        for kh in khs:
            mean = f * kh + (1 - f) * KAPPA_UNTREATED
            t = escape_time_modulated(kh, f, period)
            # relaxation time of the low phase against the time spent in it
            grid.append(dict(f=f, kappa_hi=kh, mean=float(mean),
                             predicted=bool(mean > KC), escaped=bool(np.isfinite(t)),
                             low_phase_duration=float((1 - f) * period),
                             amplitude_ratio=float(kh / KC),
                             t=None if not np.isfinite(t) else float(t)))
            print('      modulation cell %2d/%d (f=%.2f, kappa_hi=%.3f): %s'
                  % (len(grid), total, f, kh,
                     'escape' if np.isfinite(t) else 'no escape'), flush=True)
    fails = [g for g in grid if g['predicted'] != g['escaped']]

    f_fix = 0.2
    kh_fix = (1.2 * KC - (1 - f_fix) * KAPPA_UNTREATED) / f_fix
    ratios = ([1e-3, 1e-2, 0.1, 1.0] if quick else
              [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])
    sweep = []
    for j, r in enumerate(ratios, 1):
        print('      period sweep %2d/%d (T/T_escape = %g)' % (j, len(ratios), r), flush=True)
        per = r * t_ref
        t = escape_time_modulated(kh_fix, f_fix, per, tmax=6e4)
        saturated = bool(np.isfinite(t) and t < f_fix * per)   # escape inside the first high phase
        sweep.append(dict(ratio=float(r), period=float(per),
                          t=None if not np.isfinite(t) else float(t),
                          ratio_to_static=None if not np.isfinite(t) else float(t / t_ref),
                          escaped_within_first_high_phase=saturated))
    return dict(modulation_grid=grid,
                modulation_agreement=sum(1 for g in grid if g['predicted'] == g['escaped']),
                modulation_n=len(grid), modulation_failures=fails,
                modulation_period=float(period), static_escape_1p2kc=float(t_ref),
                kappa_hi_period_sweep=float(kh_fix), period_sensitivity=sweep,
                averaging_note='the field is linear in kappa, so first-order averaging is '
                               'exact; agreement is expected and only the failures are '
                               'informative')


# --------------------------------------------------------------------------
# 6. Full system: delayed completion, and the exponent with the stated start
# --------------------------------------------------------------------------
def _full_escape(kappa, Q0, s, tmax, start_kappa=None, want_annihilation=False,
                 eps=None):
    a = a_of(Q0)
    kappa_start = 0.5 * a * a / 4 if start_kappa is None else start_kappa
    b = true_branch(kappa_start, a)
    if b is None:
        return None
    y0 = np.concatenate([b, [Q0]])

    def ev_escape(t, y):
        return y[3] - a_of(min(max(y[4], 0.0), 1.0))
    ev_escape.terminal, ev_escape.direction = True, 1

    def ev_annih(t, y):
        aq = a_of(min(max(y[4], 0.0), 1.0))
        return aq * aq - 4 * kappa
    ev_annih.terminal, ev_annih.direction = False, -1

    events = [ev_escape, ev_annih] if want_annihilation else [ev_escape]
    sol = solve_ivp(full_field_factory(kappa, s, eps), (0, tmax), y0, method='LSODA',
                    rtol=1e-9, atol=1e-13, events=events)
    t = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    Qe = float(sol.y[4, -1])
    out = dict(t=t, Q_end=Qe, a_end=float(a_of(Qe)), kc_end=float(a_of(Qe) ** 2 / 4),
               kappa_start=float(kappa_start))
    if want_annihilation:
        if len(sol.t_events[1]):
            out.update(t_annihilation=float(sol.t_events[1][0]),
                       Q_annihilation=float(sol.y_events[1][0][4]),
                       Q_annihilation_closed_form=annihilation_Q(kappa))
        else:
            out.update(t_annihilation=None, Q_annihilation=None,
                       Q_annihilation_closed_form=annihilation_Q(kappa))
    return out


def analysis_full(quick=False):
    delayed = {}
    scales = [0.01] if quick else [0.01, 1.0]
    kaps = [0.003, 0.05] if quick else [0.003, 0.02, 0.05, 0.08]
    tmax = 5e7 if quick else 5e8
    for s in scales:
        for kappa in kaps:
            r = _full_escape(kappa, 0.0, s, tmax, start_kappa=kappa, want_annihilation=True)
            delayed['s=%g,kappa=%g' % (s, kappa)] = dict(escaped=r['t'] is not None, **r)

    excess = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    levels = [1.0, 0.9, 0.0] if quick else [1.0, 0.99, 0.9, 0.7, 0.5, 0.0]

    def exponent_sweep(eps):
        """Escape-time exponent of the full system at each evidence level.

        eps is passed straight through to the field.  At eps = 0 the evidence
        coordinate is frozen, so the only thing that can bend the fit is the
        fold itself; the difference between the two sweeps is the contribution
        of evidence feedback.
        """
        out = {}
        for Q0 in levels:
            a = a_of(Q0)
            kc = a * a / 4
            ts = [_full_escape(kc * (1 + e), Q0, 0.01, 1e6, eps=eps) for e in excess]
            ok = [i for i, r in enumerate(ts) if r and r['t']]
            if len(ok) < 4:
                continue
            full = -np.polyfit(np.log(excess[ok]), np.log([ts[i]['t'] for i in ok]), 1)[0]
            sm = [i for i in ok if excess[i] <= 0.05]
            small = (-np.polyfit(np.log(excess[sm]), np.log([ts[i]['t'] for i in sm]), 1)[0]
                     if len(sm) >= 3 else None)
            out['Q0=%g' % Q0] = dict(
                a=float(a), kc=float(kc), kappa_start=float(0.5 * kc), eps=float(eps),
                exponent_full_range=float(full),
                exponent_small_excess=None if small is None else float(small),
                times={str(float(excess[i])): ts[i]['t'] for i in ok})
        return out

    exps = exponent_sweep(EPS)
    control = exponent_sweep(0.0)

    # The start is 0.5 kappa_c at every threshold, and that point sits at a
    # FIXED fraction of the branch remnant, so it cannot be the source of any
    # spread across thresholds:  r1(kc/2)/r1(kc) = (1 - 1/sqrt(2)) = 0.2929.
    start_fraction = {}
    for Q0 in levels:
        a = a_of(Q0)
        kc = a * a / 4
        start_fraction['Q0=%g' % Q0] = float(first_stage(0.5 * kc, a) / (a / 2))

    # Does the control converge on the square-root law?  Push the excess two
    # decades below the grid above at two widely separated thresholds.
    deep = {}
    if not quick:
        ex = np.array([2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
        for Q0 in (0.5, 0.0):
            a = a_of(Q0)
            kc = a * a / 4
            tt = [_full_escape(kc * (1 + e), Q0, 0.01, 1e8, eps=0.0)['t'] for e in ex]
            sl = [float(-(np.log(tt[i + 1]) - np.log(tt[i]))
                        / (np.log(ex[i + 1]) - np.log(ex[i]))) for i in range(len(ex) - 1)]
            deep['Q0=%g' % Q0] = dict(excess=[float(x) for x in ex],
                                      times=[float(x) for x in tt], local_slopes=sl)

    arrest = {}
    for kappa in (0.001, 0.003, 0.005):
        lo, hi = true_branch(kappa, A0), true_branch(kappa, A1)
        arrest['%g' % kappa] = dict(profile_at_a0=[float(x) for x in lo],
                                    profile_at_a1=[float(x) for x in hi],
                                    first_stage_gain=float(hi[1] / lo[1]))
    band = {}
    for kappa in (0.006, 0.008, 0.02, 0.05, 0.08):
        band['%g' % kappa] = dict(kappa=kappa, in_band=bool(KC < kappa < KC_DEPLETED),
                                  Q_star=annihilation_Q(kappa),
                                  a_star=float(2 * np.sqrt(kappa)))
    return dict(delayed_completion=delayed, full_system_exponents=exps,
                full_system_exponents_eps0=control,
                eps0_deep_small_excess=deep,
                start_fraction_of_remnant=start_fraction,
                annihilation_locus=band, arrest_drift=arrest)


# --------------------------------------------------------------------------
# 7. Unequal coupling: the sharp quantity is the weakest stage
# --------------------------------------------------------------------------
def chain_field_vec(t, r, a, kap):
    """Forward chain with a coupling per stage rather than one shared value."""
    d = r * (1 - r) * (r - a)
    d[1:] += kap * (r[:-1] - r[1:])
    return d


def _propagate_vec(kap, a=A0, v=0.65, tmax=3e4):
    kap = np.asarray(kap, float)
    r0 = np.zeros(len(kap) + 1)
    r0[0] = v
    return solve_ivp(chain_field_vec, (0, tmax), r0, args=(a, kap), method='LSODA',
                     rtol=1e-10, atol=1e-13).y[:, -1]


def branch_vec(kap, a=A1):
    """Exact partial branch with a coupling per stage.

    Only the FIRST stage can lose its low root: for k >= 2 the cubic
    f(x) = x(1-x)(x-a) + kappa_k (r_{k-1} - x) has f(0) = kappa_k r_{k-1} > 0 and
    f(a) = -kappa_k (a - r_{k-1}) < 0 whenever 0 < r_{k-1} < a, so a low root
    always exists whatever kappa_k is.  Existence of the branch is therefore a
    statement about kappa_1 alone, and about no other coupling and no length.
    """
    r1 = first_stage(kap[0], a)
    if r1 is None:
        return None
    r = [1.0, float(r1)]
    for k in kap[1:]:
        prev = r[-1]

        def h(x, prev=prev, k=k):
            return x * (1 - x) * (x - a) + k * (prev - x)
        if h(0.0) <= 0 or h(a) >= 0:
            return None
        r.append(float(brentq(h, 0.0, a, xtol=1e-16, rtol=8.9e-16, maxiter=200)))
    return np.array(r)


def analysis_unequal(quick=False):
    # (a) each stage carries its own boundary, and it is the same number
    stage_boundaries = {}
    for stage in (1, 2, 3):
        def g(k, stage=stage):
            kap = np.full(3, 1.0)
            kap[stage - 1] = k
            return _propagate_vec(kap)[stage] - 0.5
        stage_boundaries['stage%d' % stage] = float(
            brentq(g, 1e-4, 0.5, xtol=1e-14, rtol=8.9e-16, maxiter=200))

    # (b) recovery stops at the first stage whose own coupling is subcritical
    rows = []
    grids = ([0.2, 0.2, 0.2], [0.2, 0.05, 0.2], [0.05, 0.2, 0.2], [0.2, 0.2, 0.05],
             [0.3, 0.12, 0.11], [0.5, 0.091, 0.089], [0.089, 0.5, 0.5])
    for kap in grids:
        y = _propagate_vec(kap)
        depth = int(sum(1 for x in y[1:] if x > 0.5))
        pred = next((i for i, k in enumerate(kap) if k <= KC_DEPLETED), len(kap))
        rows.append(dict(kappa=list(kap), final=[float(x) for x in y], depth=depth,
                         predicted_depth=int(pred), min_kappa=float(min(kap))))

    # (c) existence of the branch depends on kappa_1 and on nothing else
    exists = {}
    for k1 in (0.005625, 0.00563):
        for tail in ([1e-4, 1e-4], [10.0, 10.0], [0.5, 30.0]):
            exists['k1=%g,tail=%s' % (k1, tail)] = branch_vec([k1] + tail) is not None
    length = {}
    for N in range(1, 5 if quick else 13):
        lo, hi = 1e-3, 1e-2
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if branch_vec([mid] * N) is None:
                hi = mid
            else:
                lo = mid
        length['N=%d' % N] = float(lo)
    # (c') the comparison lemma that turns "r_{k-1} -> 1" into "stage k crosses
    #      in finite time".  With the predecessor held at 1 - eta the stage-k
    #      field is  G(y) = y(1-y)(y-a) + kappa (1 - eta - y),  and at eta = 0
    #      it factorises as (1-y)(y^2 - a y + kappa), which is bounded below on
    #      [0, theta] by mu = (1-theta)(kappa - a^2/4) > 0 whenever kappa > a^2/4.
    #      Since G is decreasing in eta at rate kappa, any eta < mu_theta/kappa
    #      keeps the field positive, so r_k reaches theta within theta/(mu - kappa eta).
    theta = 0.9
    xs = np.linspace(0.0, theta, 20001)
    lemma = {}
    for kappa in (0.0901, 0.095, 0.12, 0.20):
        mu_theta = float(np.min((1 - xs) * (xs ** 2 - A0 * xs + kappa)))
        mu_bound = float((1 - theta) * (kappa - KC_DEPLETED))
        eta = 0.5 * mu_theta / kappa
        sol = solve_ivp(lambda t, y, k=kappa, e=eta:
                        y * (1 - y) * (y - A0) + k * (1 - e - y),
                        (0, 1e6), [0.0], method='LSODA', rtol=1e-10, atol=1e-13,
                        events=[lambda t, y: y[0] - A0])
        t_cross = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
        lemma['%g' % kappa] = dict(
            kappa=kappa, margin=float(kappa - KC_DEPLETED), theta=theta,
            field_min_on_0_theta=mu_theta, analytic_lower_bound=mu_bound,
            eta_used=float(eta), crossing_time=t_cross,
            crossing_time_bound=float(theta / (mu_theta - kappa * eta)))

    # (d) one stage swept while its neighbours stay strong: the block sits at
    #     that stage and at no other, and it appears exactly at a0^2/4
    ks = np.concatenate([np.linspace(0.02, 0.085, 8), np.linspace(0.0875, 0.0925, 9),
                         np.linspace(0.095, 0.20, 8)])
    sweep = []
    for k2 in ks:
        y = _propagate_vec([0.2, float(k2), 0.2])
        sweep.append(dict(kappa2=float(k2), final=[float(x) for x in y]))
    return dict(stage_boundaries=stage_boundaries, arrest_depth=rows,
                stage2_sweep=sweep, induction_lemma=lemma,
                branch_exists_vs_kappa1=exists, kappa_c_vs_chain_length=length)


# --------------------------------------------------------------------------
# 8. Spectrum on the branch, and the origin of the exponent
# --------------------------------------------------------------------------
def branch_spectrum(kappa, a=A1):
    """Eigenvalues of the frozen capacity block at the partial branch.

    The block is lower triangular on a forward chain, so the spectrum is the
    diagonal, lambda_k = (1-2 r_k)(r_k - a) + r_k(1 - r_k) - kappa.  At the
    first stage, substituting kappa = a r_1 - r_1^2 collapses it exactly to
    lambda_1 = -sqrt(a^2 - 4 kappa) (1 - r_1).
    """
    b = true_branch(kappa, a)
    if b is None:
        return None
    lam = np.array([(1 - 2 * x) * (x - a) + x * (1 - x) - kappa for x in b[1:]])
    closed = -np.sqrt(max(a * a - 4 * kappa, 0.0)) * (1 - b[1])
    return lam, float(closed)


def analysis_spectrum(quick=False):
    rows, worst, top = [], 0.0, -np.inf
    fracs = [0.1, 0.5, 0.99] if quick else [1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                                            0.99, 0.999, 0.99999]
    for a in (A1, 0.30, A0):
        for f in fracs:
            kappa = f * a * a / 4
            got = branch_spectrum(kappa, a)
            if got is None:
                continue
            lam, closed = got
            worst = max(worst, abs(lam[0] - closed))
            top = max(top, float(lam.max()))
            rows.append(dict(a=float(a), frac=float(f), kappa=float(kappa),
                             eigenvalues=[float(x) for x in lam],
                             lambda1_closed_form=closed))
    least = all(abs(r['eigenvalues'][0]) <= min(abs(x) for x in r['eigenvalues'][1:]) + 1e-15
                for r in rows)
    gap = min(abs(r['eigenvalues'][1]) - abs(r['eigenvalues'][0]) for r in rows)
    return dict(spectrum=rows, spectrum_max_deviation=float(worst),
                spectrum_largest_eigenvalue=float(top),
                spectrum_lambda1_is_least_negative=bool(least),
                spectrum_smallest_gap=float(gap))


# --------------------------------------------------------------------------
# 9. How deep the well is, and what a little noise does to it
# --------------------------------------------------------------------------
def well_depth(kappa, a=A1):
    """Barrier of the first-stage well with the root held at 1.

    dr1/dt = -dV/dr1 with V(r) = -∫ [ r(1-r)(r-a) + kappa(1-r) ] dr, so the
    barrier is V at the unstable root minus V at the stable one.
    """
    d = a * a - 4 * kappa
    if d <= 0:
        return None
    D = np.sqrt(d)
    rs, ru = (a - D) / 2, (a + D) / 2

    def V(r):
        return -(-r ** 4 / 4 + (1 + a) * r ** 3 / 3 - a * r ** 2 / 2
                 + kappa * (r - r ** 2 / 2))
    return dict(kappa=float(kappa), barrier=float(V(ru) - V(rs)),
                r_stable=float(rs), r_unstable=float(ru))


def sde_escape(kappa, sigma, npath=500, tmax=2e4, dt=0.05, a=A1, seed=SEED):
    """Euler-Maruyama on the capacity block with the root held at 1.

    Escape is the first stage crossing its own unstable root.  Returns the
    escape time of each path, nan for the paths that never left.
    """
    rng = np.random.default_rng(seed)
    b = true_branch(kappa, a)
    r = np.tile(b[1:], (npath, 1)).copy()
    ru = (a + np.sqrt(a * a - 4 * kappa)) / 2
    esc = np.full(npath, np.nan)
    s = sigma * np.sqrt(dt)
    for i in range(int(tmax / dt)):
        d = r * (1 - r) * (r - a)
        d[:, 0] += kappa * (1.0 - r[:, 0])
        d[:, 1:] += kappa * (r[:, :-1] - r[:, 1:])
        r = np.clip(r + d * dt + s * rng.standard_normal(r.shape), 0.0, 1.0)
        m = np.isnan(esc) & (r[:, 0] > ru)
        esc[m] = (i + 1) * dt
        if not np.isnan(esc).any():
            break
    return esc


def analysis_noise(quick=False):
    wells = {'%g' % k: well_depth(k) for k in (0.001, 0.003, 0.005, 0.0055)}
    sigmas = (0.004, 0.008) if quick else (0.002, 0.004, 0.006, 0.008)
    npath = 120 if quick else 500
    tmax = 4e3 if quick else 2e4
    rows = {}
    for sig in sigmas:
        e = sde_escape(KAPPA_UNTREATED, sig, npath=npath, tmax=tmax)
        n = int(np.sum(~np.isnan(e)))
        rows['%g' % sig] = dict(sigma=float(sig), paths=int(npath), escaped=n,
                                horizon=float(tmax),
                                median=None if n == 0 else float(np.nanmedian(e)),
                                iqr=None if n == 0 else
                                [float(np.nanpercentile(e, 25)),
                                 float(np.nanpercentile(e, 75))])
        grid = np.linspace(0, tmax, 200)
        rows['%g' % sig]['survival'] = [
            float(np.mean(np.isnan(e) | (e > t))) for t in grid]
        rows['%g' % sig]['survival_grid'] = [float(t) for t in grid]
    # Halving the step leaves the escaped fraction alone and moves the median
    # by more than the digits a two-figure quotation would imply, which is why
    # the text reports the median only to an order of magnitude.
    control = None
    if not quick:
        e = sde_escape(KAPPA_UNTREATED, 0.008, npath=npath, tmax=tmax, dt=0.025)
        n = int(np.sum(~np.isnan(e)))
        control = dict(sigma=0.008, dt=0.025, paths=int(npath), escaped=n,
                       median=None if n == 0 else float(np.nanmedian(e)))
    return dict(well_depth=wells, noise_escape=rows, noise_step_control=control)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
# Okabe-Ito subset, validated for colour-vision deficiency; fixed order, never cycled.
PALETTE = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
INK, MUTED = '#1a1a1a', '#5b5b5b'
DASH = ['-', '--', ':', '-.']
MARK = ['o', 's', '^', 'D']


def _style(plt):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7,
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK,
        'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'legend.frameon': False, 'legend.fontsize': 6.5,
        'lines.linewidth': 1.4, 'savefig.dpi': 400, 'figure.dpi': 200,
        'text.color': INK,
    })


SOURCE_DATA = {}          # panel -> {'columns': [...], 'rows': [[...], ...]}


def make_figures(D):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    _style(plt)
    C = PALETTE
    rng = np.random.default_rng(SEED + 1)

    def tidy(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def panel(ax, letter):
        ax.set_title(letter, loc='left', pad=6, fontsize=8.5, fontweight='bold', color=INK)

    # ------------------------------------------------------------- Figure 1
    SOURCE_DATA.clear()
    fig = plt.figure(figsize=(7.09, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.34,
                          left=0.085, right=0.985, top=0.94, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a')
    ax.set_xlim(-0.6, 4.2); ax.set_ylim(-1.30, 1.35); ax.axis('off')
    xs = [0, 1.1, 2.2, 3.3]
    for i, x in enumerate(xs):
        ax.add_patch(Rectangle((x - 0.30, -0.28), 0.60, 0.56, facecolor='white',
                               edgecolor=INK, linewidth=0.9))
        ax.text(x, 0, '$r_%d$' % i, ha='center', va='center', fontsize=8)
        ax.text(x, -0.46, ['sensory', 'policy', 'motiv.', 'fast vol.'][i],
                ha='center', va='top', fontsize=5.6, color=MUTED)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 0.31, 0), (xs[i + 1] - 0.31, 0),
                                         arrowstyle='-|>', mutation_scale=8,
                                         linewidth=1.2, color=C[1]))
            ax.text((x + xs[i + 1]) / 2, 0.16, r'$\kappa$', ha='center',
                    fontsize=7.5, color=C[1])
    ax.add_patch(FancyArrowPatch((0, 0.95), (0, 0.31), arrowstyle='-|>',
                                 mutation_scale=8, linewidth=1.2, color=C[2]))
    ax.text(0.10, 1.02, r'$u(t)$  root excursion', ha='left', va='center',
            fontsize=7, color=C[2])
    ax.text(1.65, -0.80, r'crossing set by $u$   |   propagation set by $\kappa$',
            ha='center', fontsize=6.4, color=MUTED)
    ax.text(1.65, -1.12, 'while every capacity is below threshold no non-negative\n'
                         'coupling can cross, on this or any other graph',
            ha='center', fontsize=6.2, color=MUTED, style='italic')

    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
    fams = [r['family'] for r in D['max_principle']]
    b_cols, b_rows = ['time'], None
    for j, fam in enumerate(fams):
        n = int(rng.integers(3, 7))
        W = _random_weights(rng, n, fam)
        r0 = rng.random(n) * A0 * 0.999
        ts = np.linspace(0, 120, 400)
        y = solve_ivp(graph_field, (0, 120), r0, args=(A0, W), method='LSODA',
                      rtol=1e-9, atol=1e-12, dense_output=True).sol(ts)
        ax.plot(ts, y.max(0), DASH[j], color=C[j], linewidth=1.3, label=fam)
        b_cols.append('max_k r_k, %s' % fam)
        col = y.max(0)
        b_rows = [[t] for t in ts] if b_rows is None else b_rows
        for row, v in zip(b_rows, col):
            row.append(float(v))
    SOURCE_DATA['Fig 1b'] = dict(columns=b_cols, rows=b_rows)
    ax.axhline(A0, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(2, A0 + 0.02, r'threshold $a_0$', fontsize=6.3)
    ax.set_xlabel('time'); ax.set_ylabel(r'$\max_k r_k(t)$')
    ax.set_ylim(0, 0.72)
    ax.legend(loc='center right', handletextpad=0.4, labelspacing=0.2)
    ax.text(0.97, 0.06, '%d of %d runs reached $a_0$;\nthe maximum never rose (%.0e)'
            % (D['max_principle_total_crossings'], D['max_principle_total_trials'],
               max(r['max_rise_of_maximum'] for r in D['max_principle'])),
            transform=ax.transAxes, fontsize=6.2, ha='right', color=INK)
    tidy(ax)

    ax = fig.add_subplot(gs[1, 0]); panel(ax, 'c')
    rows = D['pulse_rows']
    taus = sorted({r['tau'] for r in rows})
    for j, tau in enumerate(taus):
        sub = [r for r in rows if r['tau'] == tau]
        for r in sub:
            ax.plot(r['root_at_pulse_end'], j, marker='o' if r['recovered'] else 'x',
                    color=C[0] if r['recovered'] else C[2],
                    markersize=4.4 if r['recovered'] else 4.0,
                    markeredgewidth=0 if r['recovered'] else 1.1, linestyle='none')
    ax.axvline(A0, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(A0 - 0.006, len(taus) - 0.55, r'$a_0=0.60$', rotation=90, ha='right',
            va='top', fontsize=6.3)
    ax.set_yticks(range(len(taus)))
    ax.set_yticklabels([r'$\tau=%g$' % t for t in taus])
    ax.set_xlabel(r'root value at the end of the pulse,  $r_0(\tau)$')
    ax.set_ylabel('pulse duration')
    ax.set_ylim(-0.85, len(taus) - 0.35)
    ax.plot([], [], 'o', color=C[0], markersize=4.4, markeredgewidth=0, label='recovers')
    ax.plot([], [], 'x', color=C[2], markersize=4.0, markeredgewidth=1.1, label='collapses')
    ax.legend(loc='lower right', handletextpad=0.3, labelspacing=0.2, ncol=2,
              columnspacing=1.0, borderpad=0.1)
    tidy(ax)

    ax = fig.add_subplot(gs[1, 1]); panel(ax, 'd')
    prop = D['propagation']
    keys = sorted(prop, key=float)
    ks = np.array([float(k) for k in keys])
    ax.semilogx(ks, [prop[k][3] for k in keys], '-', color=C[0], marker='o',
                markersize=3.4, markeredgewidth=0)
    ax.semilogx(ks, [prop[k][1] for k in keys], '--', color=C[2], marker='s',
                markersize=3.0, markeredgewidth=0)
    ax.annotate('$r_3$', xy=(ks[-1], prop[keys[-1]][3]), xytext=(3, -1),
                textcoords='offset points', fontsize=6.8, color=C[0])
    ax.annotate('$r_1$', xy=(ks[0], prop[keys[0]][1]), xytext=(2, 5),
                textcoords='offset points', fontsize=6.8, color=C[2])
    ax.axvline(KC_DEPLETED, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(KC_DEPLETED * 0.93, 0.55, r'$\kappa_c=a_0^{2}/4$', fontsize=6.3,
            rotation=90, va='center', ha='right')
    ax.set_xlabel(r'coupling  $\kappa$')
    ax.set_ylabel('final state after the crossing')
    ax.set_ylim(-0.06, 1.12)
    ax.text(0.03, 0.90, 'root crossed at $v=0.65$ in every run',
            transform=ax.transAxes, fontsize=6.3)
    ax.text(0.03, 0.80, 'boundary located at %.7f' % D['propagation_boundary'],
            transform=ax.transAxes, fontsize=6.3, color=MUTED)
    tidy(ax)

    fig.savefig('Fig1.png', bbox_inches='tight', facecolor='white')
    fig.savefig('Fig1.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ------------------------------------------------------------- Figure 2
    fig = plt.figure(figsize=(7.09, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.44, left=0.075, right=0.985, top=0.85,
                          bottom=0.23)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a')
    kk = np.linspace(1e-5, KC * 0.9995, 240)
    TB = np.array([true_branch(k) for k in kk])
    CB = np.array([closed_branch(k) for k in kk])
    SOURCE_DATA['Fig 2a'] = dict(
        columns=['kappa', 'r1 exact', 'r2 exact', 'r3 exact',
                 'r1 closed form', 'r2 closed form', 'r3 closed form',
                 'r2 closed-form error (%)', 'r3 closed-form error (%)'],
        rows=[[float(k)] + [float(x) for x in TB[i, 1:4]] + [float(x) for x in CB[i, 1:4]]
              + [float(100 * (CB[i, j] - TB[i, j]) / TB[i, j]) for j in (2, 3)]
              for i, k in enumerate(kk)])
    for i in range(1, 4):
        ax.semilogy(kk, TB[:, i], DASH[i - 1], color=C[i - 1], linewidth=1.4)
        ax.semilogy(kk, CB[:, i], color=MUTED, linewidth=0.7, alpha=0.9)
        ax.annotate('$r_%d$' % i, xy=(kk[-1], TB[-1, i]), xytext=(3, 0),
                    textcoords='offset points', fontsize=6.8, color=C[i - 1], va='center')
    ax.axvline(KC, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.text(KC * 0.97, 2e-4, r'$\kappa_c=a_1^{2}/4$', rotation=90, ha='right',
            va='center', fontsize=6.3)
    ax.set_xlabel(r'coupling  $\kappa$'); ax.set_ylabel('partial-branch capacity')
    ax.set_xlim(0, KC * 1.1)
    ax.text(0.03, 0.02, 'colour: exact equilibrium\ngrey: stage-wise closed form',
            transform=ax.transAxes, fontsize=5.8, color=INK)
    tidy(ax)
    # inset: the closed form is exact at stage 1 and not afterwards
    ins = ax.inset_axes([0.57, 0.15, 0.40, 0.31])
    err = 100 * (CB[:, 1:4] - TB[:, 1:4]) / TB[:, 1:4]
    for i in range(3):
        ins.plot(kk, err[:, i], DASH[i], color=C[i], linewidth=1.1)
    ins.axhline(0.0, color=INK, linewidth=0.6, linestyle=(0, (2, 2)))
    ins.set_xlim(0, KC * 1.1)
    ins.set_ylim(-0.6, max(6.0, float(np.nanmax(err)) * 1.15))
    ins.set_xticks([])
    ins.tick_params(labelsize=5.4, width=0.5, pad=1.5)
    ins.text(0.02, 1.04, '% above exact', transform=ins.transAxes,
             fontsize=5.4, color=INK)
    for sp in ('top', 'right'):
        ins.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ins.spines[sp].set_linewidth(0.5)
    ins.annotate('$r_1$: 0', xy=(KC * 0.55, 0.0), xytext=(0, 3.2),
                 textcoords='offset points', fontsize=5.4, color=C[0])
    ins.annotate('$r_3$', xy=(kk[-1], err[-1, 2]), xytext=(-9, -1),
                 textcoords='offset points', fontsize=5.4, color=C[2])
    ins.annotate('$r_2$', xy=(kk[-1], err[-1, 1]), xytext=(-9, -1),
                 textcoords='offset points', fontsize=5.4, color=C[1])

    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
    aa = np.linspace(A1, A0, 300)
    SOURCE_DATA['Fig 2b'] = dict(
        columns=['threshold a(Q)', 'kappa_c = a^2/4', 'evidence Q* read across'],
        rows=[[float(a), float(a * a / 4),
               float(((A0 - a) / (A0 - A1)) ** (1.0 / P_EXP))] for a in aa])
    ax.fill_between([A1, A0], 0, KC, color=C[2], alpha=0.13, linewidth=0)
    ax.fill_between([A1, A0], KC, KC_DEPLETED, color=C[1], alpha=0.13, linewidth=0)
    ax.fill_between([A1, A0], KC_DEPLETED, KC_DEPLETED * 1.35, color=C[0],
                    alpha=0.13, linewidth=0)
    ax.plot(aa, aa ** 2 / 4, color=INK, linewidth=1.3)
    ax.axhline(KC, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax.axhline(KC_DEPLETED, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
    ax.text(0.585, KC * 0.45, 'confined to the branch', fontsize=6.0, ha='right', color=INK)
    ax.text(0.163, KC + (KC_DEPLETED - KC) * 0.80, 'stalls, completes later',
            fontsize=6.0, ha='left', color=INK)
    ax.text(0.585, KC_DEPLETED * 1.16, 'propagates at once', fontsize=6.0,
            ha='right', color=INK)
    ax.set_xlabel(r'threshold  $a(Q)$'); ax.set_ylabel(r'coupling  $\kappa$')
    ax.set_xlim(A1, A0); ax.set_ylim(0, KC_DEPLETED * 1.35)
    ax.set_yticks([0, KC, 0.05, KC_DEPLETED])
    ax.set_yticklabels(['0', '0.0056', '0.05', '0.090'])
    ax.set_xticks([A1, 0.3, 0.45, A0])

    def a_to_Q(a):
        return ((A0 - np.clip(np.asarray(a, float), A1, A0)) / (A0 - A1)) ** (1.0 / P_EXP)

    def Q_to_a(Q):
        return A0 - (A0 - A1) * np.clip(np.asarray(Q, float), 0.0, 1.0) ** P_EXP

    sec = ax.secondary_xaxis('top', functions=(a_to_Q, Q_to_a))
    sec.set_xlabel('evidence  $Q$', fontsize=6.5, labelpad=2)
    sec.tick_params(labelsize=6.0, colors=MUTED, width=0.6)
    sec.set_xticks([0.0, 0.4, 0.6, 0.8, 0.95])
    for kap in (0.02, 0.05, 0.08):
        ax.plot([2 * np.sqrt(kap)], [kap], 'o', color=INK, markersize=3.2,
                markeredgewidth=0, zorder=5)
    ax.annotate(r'$\kappa_c(a)=a^{2}/4$,  read across: $Q^{*}(\kappa)$',
                xy=(2 * np.sqrt(0.05), 0.05), xytext=(-8, -24),
                textcoords='offset points', fontsize=6.0, color=INK, ha='right',
                arrowprops=dict(arrowstyle='-', color=INK, linewidth=0.5))

    ax = fig.add_subplot(gs[0, 2]); panel(ax, 'c')
    noise = D['identification_noise']
    pcts = sorted(noise, key=float)
    xs2 = np.arange(len(pcts) + 1)
    bias = D['identification_bias']
    b0 = [b for b in bias if abs(b['kappa'] - KAPPA_UNTREATED) < 1e-12][0]
    two_med = [1 + b0['kappa_rel_error']] + [noise[p]['two_ratio']['k_median'] / KAPPA_UNTREATED
                                             for p in pcts]
    two_lo = [1 + b0['kappa_rel_error']] + [noise[p]['two_ratio']['k_iqr'][0] / KAPPA_UNTREATED
                                            for p in pcts]
    two_hi = [1 + b0['kappa_rel_error']] + [noise[p]['two_ratio']['k_iqr'][1] / KAPPA_UNTREATED
                                            for p in pcts]
    one_med = [1.0] + [noise[p]['primary']['k_median'] / KAPPA_UNTREATED for p in pcts]
    one_lo = [1.0] + [noise[p]['primary']['k_iqr'][0] / KAPPA_UNTREATED for p in pcts]
    one_hi = [1.0] + [noise[p]['primary']['k_iqr'][1] / KAPPA_UNTREATED for p in pcts]
    two_med, two_lo, two_hi = map(np.array, (two_med, two_lo, two_hi))
    one_med, one_lo, one_hi = map(np.array, (one_med, one_lo, one_hi))
    ax.errorbar(xs2 - 0.10, two_med, yerr=[two_med - two_lo, two_hi - two_med],
                fmt='o', color=C[2], markersize=4, capsize=2.5, elinewidth=1,
                markeredgewidth=0, label='two stage ratios')
    ax.errorbar(xs2 + 0.10, one_med, yerr=[one_med - one_lo, one_hi - one_med],
                fmt='s', color=C[0], markersize=4, capsize=2.5, elinewidth=1,
                markeredgewidth=0, label=r'$\kappa=a r_1-r_1^{2}$')
    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_xticks(xs2)
    ax.set_xticklabels(['0'] + ['%g%%' % (float(p) * 100) for p in pcts])
    ax.set_xlabel('lognormal noise on the stage ratios')
    ax.set_ylabel(r'$\hat\kappa$ / true value')
    ax.legend(loc='lower left', handletextpad=0.3, labelspacing=0.2)
    ax.text(0.03, 0.93, 'evaluated on the exact equilibrium', transform=ax.transAxes,
            fontsize=6.0, color=MUTED)
    tidy(ax)

    fig.savefig('Fig2.png', bbox_inches='tight', facecolor='white')
    fig.savefig('Fig2.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ------------------------------------------------------------- Figure 3
    fig = plt.figure(figsize=(7.09, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.075, right=0.985, top=0.85,
                          bottom=0.23)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a')
    sw = D['stage2_sweep']
    k2 = [s['kappa2'] for s in sw]
    for i in range(1, 4):
        ax.plot(k2, [s['final'][i] for s in sw], DASH[i - 1], color=C[i - 1],
                linewidth=1.4, label=r'$r_%d$' % i)
    ax.axvline(KC_DEPLETED, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.text(KC_DEPLETED * 1.06, 0.60, r'$a_0^{2}/4$', fontsize=6.3)
    ax.set_xlabel(r'coupling of the second stage  $\kappa_2$')
    ax.set_ylabel('final capacity')
    ax.set_ylim(-0.06, 1.12)
    ax.text(0.04, 0.70, r'$\kappa_1=\kappa_3=0.2$',
            transform=ax.transAxes, fontsize=6.0, color=MUTED)
    ax.legend(loc=(0.04, 0.30), handletextpad=0.3, labelspacing=0.2,
              borderpad=0.15, frameon=False)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
    NE = D['noise_escape']
    for i, key in enumerate(sorted(NE, key=float)):
        v = NE[key]
        ax.plot(v['survival_grid'], v['survival'], DASH[i % 4], color=C[i % 4],
                linewidth=1.3, label=r'$\sigma=%s$' % key)
    ax.set_xlabel('time'); ax.set_ylabel('fraction still on the branch')
    ax.set_ylim(-0.04, 1.06)
    ax.legend(loc='center right', handletextpad=0.3, labelspacing=0.2, borderpad=0.15,
              frameon=False)
    w = D['well_depth']['0.003']
    ax.text(0.03, 0.06, 'barrier %.1e at $\\kappa=0.003$' % w['barrier'],
            transform=ax.transAxes, fontsize=6.0, color=MUTED)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 2]); panel(ax, 'c')
    S = [s for s in D['spectrum'] if abs(s['a'] - A1) < 1e-12]
    x = [1 - s['frac'] for s in S]
    ax.loglog(x, [-s['eigenvalues'][0] for s in S], 'o', color=C[0], markersize=4,
              markeredgewidth=0, label=r'$-\lambda_1$, numerical')
    xx = np.logspace(-5.2, -0.001, 300)
    lam = [np.sqrt(A1 ** 2 * t) * (1 - first_stage((1 - t) * A1 ** 2 / 4, A1)) for t in xx]
    ax.loglog(xx, lam, color=INK, linewidth=1.0,
              label=r'$\sqrt{a^{2}-4\kappa}\,(1-r_1)$')
    ax.loglog(x, [-s['eigenvalues'][1] for s in S], 's', color=C[1], markersize=3.4,
              markeredgewidth=0, label=r'$-\lambda_2$')
    ax.set_xlabel(r'distance from the fold  $1-\kappa/\kappa_c$')
    ax.set_ylabel('decay rate on the branch')
    ax.legend(loc='lower right', handletextpad=0.3, labelspacing=0.25,
              borderpad=0.15, frameon=False)
    ax.text(0.06, 0.55, r'slope $1/2$', transform=ax.transAxes, fontsize=6.3, color=INK)
    tidy(ax)

    fig.savefig('Fig3.png', bbox_inches='tight', facecolor='white')
    fig.savefig('Fig3.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ------------------------------------------------------------- Figure 4
    fig = plt.figure(figsize=(7.09, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.36,
                          left=0.09, right=0.985, top=0.94, bottom=0.09)

    ax = fig.add_subplot(gs[0, 0]); panel(ax, 'a')
    for r in D['modulation_grid']:
        ok = r['escaped']
        ax.plot(r['f'], r['kappa_hi'], marker='o' if ok else 'x',
                color=C[0] if ok else C[2], markersize=4.4 if ok else 4.0,
                markeredgewidth=0 if ok else 1.1, linestyle='none',
                markerfacecolor=C[0] if ok else 'none')
    ff = np.linspace(0.03, 0.75, 200)
    ax.plot(ff, (KC - (1 - ff) * KAPPA_UNTREATED) / ff, color=INK, linewidth=1.0)
    ax.set_yscale('log'); ax.set_ylim(0.0042, 0.65); ax.set_xlim(-0.02, 0.80)
    ax.set_xlabel('duty fraction  $f$')
    ax.set_ylabel(r'elevated coupling  $\kappa_{\rm high}$')
    ax.text(0.02, 0.955, r'$f\kappa_{\rm high}+(1-f)\kappa_{\rm low}=\kappa_c$',
            transform=ax.transAxes, fontsize=6.5, va='top')
    ax.plot([], [], 'o', color=C[0], markersize=4.4, markeredgewidth=0, label='escape')
    ax.plot([], [], 'x', color=C[2], markersize=4.0, markeredgewidth=1.1, label='no escape')
    ax.legend(loc='upper right', handletextpad=0.3, labelspacing=0.2, ncol=2,
              columnspacing=1.0, borderpad=0.1)
    ax.annotate('averaging overstates efficacy here', xy=(0.055, 0.115),
                xytext=(0.17, 0.30), fontsize=6.0, color=C[2], va='center',
                arrowprops=dict(arrowstyle='->', color=C[2], linewidth=0.6,
                                connectionstyle='arc3,rad=-0.25'))
    tidy(ax)

    ax = fig.add_subplot(gs[0, 1]); panel(ax, 'b')
    P = D['period_sensitivity']
    x = [p['ratio'] for p in P]
    y = [p['ratio_to_static'] if p['ratio_to_static'] else np.nan for p in P]
    sat = [p.get('escaped_within_first_high_phase') for p in P]
    ax.loglog(x, y, '-', color=C[0], linewidth=1.3)
    for xi, yi, si in zip(x, y, sat):
        ax.plot(xi, yi, marker='D' if si else 'o', color=C[0] if not si else C[2],
                markersize=3.8, markeredgewidth=0)
    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (2, 2)))
    ax.text(x[0] * 1.1, 1.06, 'averaging prediction', fontsize=6.3, color=MUTED)
    ax.text(0.03, 0.10, 'below the line: the modulated system escapes\nsooner than the '
                        'average predicts', transform=ax.transAxes, fontsize=6.0, color=INK)
    ax.text(0.03, 0.26, 'diamonds: escape completes\nwithin the first elevated phase',
            transform=ax.transAxes, fontsize=6.0, color=C[2])
    ax.set_xlabel('modulation period / static escape time')
    ax.set_ylabel('escape time / static escape time')
    tidy(ax)

    ax = fig.add_subplot(gs[1, 0]); panel(ax, 'c')
    E = D['escape_times']
    ex = np.array([e['excess'] for e in E])
    sl = np.array([e['local_slope'] if e['local_slope'] else np.nan for e in E])
    mid = np.sqrt(ex[1:] * ex[:-1])
    ax.semilogx(mid, sl[1:], '-', color=C[0], marker='o', markersize=3.6,
                markeredgewidth=0)
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(mid[0], 0.508, r'$1/2$', fontsize=6.5)
    ax.set_xlabel(r'excess  $(\kappa-\kappa_c)/\kappa_c$')
    ax.set_ylabel('local slope of  log T  vs  log excess')
    ax.set_ylim(0.48, 0.68)
    ax.text(0.04, 0.28, 'slope %.4f at the smallest excess' % E[1]['local_slope'],
            transform=ax.transAxes, fontsize=6.3)
    tidy(ax)
    fx, f0 = D.get('full_system_exponents', {}), D.get('full_system_exponents_eps0', {})
    if fx and f0:
        ins = ax.inset_axes([0.10, 0.50, 0.45, 0.45])
        keys = sorted(set(fx) & set(f0), key=lambda k: fx[k]['a'])
        aa = [fx[k]['a'] for k in keys]
        ins.plot(aa, [f0[k]['exponent_small_excess'] for k in keys], 'o-', color=MUTED,
                 markersize=3.0, markeredgewidth=0, linewidth=1.0,
                 label=r'$\varepsilon=0$')
        ins.plot(aa, [fx[k]['exponent_small_excess'] for k in keys], 's-', color=C[2],
                 markersize=3.0, markeredgewidth=0, linewidth=1.0,
                 label=r'$\varepsilon=0.02$')
        ins.set_ylim(0.34, 0.58)
        ins.set_xlabel(r'threshold  $a(Q_0)$', fontsize=5.6, labelpad=1.2)
        ins.text(0.0, 1.05, 'exponent fitted below 5% excess', transform=ins.transAxes,
                 fontsize=5.4, color=INK)
        ins.tick_params(labelsize=5.4, width=0.5, pad=1.5)
        ins.legend(fontsize=5.2, loc='lower left', handlelength=1.4, handletextpad=0.3,
                   labelspacing=0.15, borderpad=0.15, frameon=False)
        for sp in ('top', 'right'):
            ins.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ins.spines[sp].set_linewidth(0.5)

    ax = fig.add_subplot(gs[1, 1]); panel(ax, 'd')
    traj = D['staircase'].get('trajectory')
    S = np.asarray(traj, float).T if traj is not None else np.load('staircase.npy')
    ts, ys = S[0], S[1:]
    for i in range(1, 4):
        ax.plot(ts / ts[-1], ys[i], DASH[i - 1], color=C[i - 1], linewidth=1.4)
        ax.annotate('$r_%d$' % i, xy=(1.0, ys[i][-1]), xytext=(4, [5, -5, 7][i - 1]),
                    textcoords='offset points', fontsize=6.8, color=C[i - 1], va='center')
    ax.axhline(A1, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    ax.text(0.02, A1 + 0.02, r'$a_1$', fontsize=6.3, color=MUTED)
    ax.set_xlabel('time / escape time'); ax.set_ylabel('capacity')
    ax.set_xlim(0, 1.10); ax.set_ylim(-0.05, 1.10)
    tr = D['staircase']['transit_a1_to_0p9']
    ax.text(0.03, 0.93, r'$r_1$ and $r_2$ cross $a_1$ and reach 0.9 in'
                        '\n%.1f%% of the interval in both cases' % (100 * tr['r1']),
            transform=ax.transAxes, fontsize=6.2, color=INK)
    ax.text(0.03, 0.76, r'$r_3$ crosses at 1.00 by construction', transform=ax.transAxes,
            fontsize=6.2, color=MUTED)
    tidy(ax)

    fig.savefig('Fig4.png', bbox_inches='tight', facecolor='white')
    fig.savefig('Fig4.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)



# --------------------------------------------------------------------------
# 10. What the confinement theorem does NOT cover
# --------------------------------------------------------------------------
def additive_field(t, r, a, w, kind):
    """Same bistable law, but coupling enters as a non-negative additive drive.

    dr_i/dt = f(r_i) + w (1 - r_i) sum_{j != i} S(r_j),  S >= 0, complete graph.
    The gate (1 - r_i) keeps the state in [0,1]; the drive has no difference
    structure, so the sign argument behind the maximum principle does not apply.
    """
    S = r if kind == 'linear' else 1.0 / (1.0 + np.exp(-(r - 0.30) / 0.05))
    return r * (1 - r) * (r - a) + w * (1 - r) * (S.sum() - S)


def _additive_final(w, kind, r0=0.30, a=A0, n=4, tmax=4000.0):
    y = solve_ivp(additive_field, (0, tmax), np.full(n, r0), args=(a, w, kind),
                  method='LSODA', rtol=1e-10, atol=1e-13).y[:, -1]
    return y


def analysis_coupling_form(quick=False):
    """The scope of the confinement theorem, tested against three escapes.

    (a) additive excitatory coupling converts a wholly sub-threshold chain with
        no external input at all, which diffusive coupling never does;
    (b) lowering the threshold converts it as well, so the list of externally
        accessible quantities is a modelling choice and not a theorem;
    (c) a crossing at the root propagates only above a0^2/4, so 'the root above
        threshold converts the chain' needs the supercritical qualifier.
    """
    add = {}
    for kind in ('linear', 'sigmoid'):
        rows = []
        for w in (0.0, 0.03, 0.05, 0.08, 0.10, 0.15):
            y = _additive_final(w, kind)
            rows.append(dict(w=float(w), final=[float(x) for x in y],
                             crossed=bool(y.max() > A0)))

        def g(w, kind=kind):
            return _additive_final(w, kind).max() - 0.5
        add[kind] = dict(rows=rows,
                         critical_w=float(brentq(g, 0.01, 0.5, xtol=1e-12)))

    # the same start under diffusive coupling, at weights three orders larger
    W = np.ones((4, 4)) - np.eye(4)

    def dif(t, r, a, w):
        return r * (1 - r) * (r - a) + w * (W * (r[None, :] - r[:, None])).sum(1)
    diffusive = []
    for w in (0.1, 1.0, 10.0, 100.0):
        y = solve_ivp(dif, (0, 4000), np.full(4, 0.30), args=(A0, w), method='LSODA',
                      rtol=1e-10, atol=1e-13).y[:, -1]
        diffusive.append(dict(w=float(w), final=[float(x) for x in y],
                              crossed=bool(y.max() > A0)))

    # lowering the threshold, with no input and no change of coupling
    thresh = []
    for a in (0.60, 0.40, 0.30, 0.25):
        y = integrate(np.full(4, 0.30), a, KAPPA_REF, 4000.0).y[:, -1]
        thresh.append(dict(a=float(a), final=[float(x) for x in y],
                           recovered=bool(y[3] > 0.5)))

    # a root left above threshold propagates only above a0^2/4
    prop = []
    for kappa in (0.6, 0.2, 0.095, 0.085, 0.05, 0.01):
        r0 = np.zeros(4)
        r0[0] = 0.61
        y = integrate(r0, A0, kappa, 3e4).y[:, -1]
        prop.append(dict(kappa=float(kappa), final=[float(x) for x in y],
                         chain_recovered=bool(y[3] > 0.5),
                         supercritical=bool(kappa > KC_DEPLETED)))
    return dict(additive_coupling=add, diffusive_control=diffusive,
                threshold_intervention=thresh, root_above_threshold=prop)


# --------------------------------------------------------------------------
# 11. The intermediate band is stage-wise too
# --------------------------------------------------------------------------
def analysis_stagewise_band(quick=False):
    """Delayed completion with a coupling per stage.

    Each stage has its own annihilation locus Q*(kappa_k), but a stage cannot
    cross before its predecessor, so the level at which stage k completes is the
    RUNNING MAXIMUM of Q* over the stages up to k, not Q*(kappa_k) alone.  A
    stage whose own locus lies below that running maximum is released by its
    predecessor and follows it almost at once.
    """
    vecs = ([[0.5, 0.05, 0.02], [0.5, 0.02, 0.05]] if quick else
            [[0.5, 0.05, 0.02], [0.5, 0.02, 0.05], [0.2, 0.08, 0.03],
             [0.3, 0.04, 0.07], [0.5, 0.06, 0.015]])
    tmax = 5e6 if quick else 5e8
    out = []
    for kap in vecs:
        events = []
        for k in (1, 2, 3):
            def ev(t, y, k=k):
                return y[k] - a_of(min(max(y[4], 0.0), 1.0))
            ev.terminal, ev.direction = False, 1
            events.append(ev)
        sol = solve_ivp(full_field_vec_factory(kap, 0.01), (0, tmax),
                        [1.0, 0.0, 0.0, 0.0, 0.0], method='LSODA',
                        rtol=1e-9, atol=1e-13, events=events)
        running, rows = 0.0, []
        for k in (1, 2, 3):
            qs = annihilation_Q(kap[k - 1])
            qs = 0.0 if qs is None else float(qs)
            running = max(running, qs)          # a stage waits for its predecessor
            te, ye = sol.t_events[k - 1], sol.y_events[k - 1]
            rows.append(dict(
                stage=k, kappa=float(kap[k - 1]), Q_star_own=qs,
                Q_star_running_max=float(running),
                t=float(te[0]) if len(te) else None,
                Q=float(ye[0][4]) if len(te) else None,
                overshoot=float(ye[0][4] - running) if len(te) else None,
                gated_by_predecessor=bool(qs < running - 1e-12)))
        ts = [r['t'] for r in rows if r['t'] is not None]
        gap = None
        if len(ts) >= 3:
            gap = float((ts[2] - ts[1]) / ts[1])
        out.append(dict(kappa=list(map(float, kap)), stages=rows,
                        second_gap_relative=gap))
    ok = [r for v in out for r in v['stages'] if r['overshoot'] is not None]
    return dict(stagewise_band=out,
                stagewise_overshoot_max=float(max(r['overshoot'] for r in ok)),
                stagewise_overshoot_min=float(min(r['overshoot'] for r in ok)),
                stagewise_events=len(ok))


# --------------------------------------------------------------------------
# Source data: one sheet per figure panel, as required with the submission
# --------------------------------------------------------------------------
def source_data_tables(D):
    """Assemble the plotted values of every panel, panel by panel.

    Panels whose curves are computed inside make_figures are captured there in
    SOURCE_DATA; the rest are read straight out of results.json, so the tables
    are the numbers the figures were drawn from and not a second computation.
    """
    T = {}
    T['Read me'] = dict(columns=['item', 'value'], rows=[
        ['Manuscript', MANUSCRIPT_TITLE],
        ['Source data for', 'Figs. 1 to 4'],
        ['Generated by', 'saito_two_axes_reproduce.py, from results.json'],
        ['a0', A0], ['a1', A1], ['p', P_EXP], ['epsilon', EPS], ['rho', RHO],
        ['illustrative kappa', KAPPA_REF], ['partial-branch kappa', KAPPA_UNTREATED],
        ['kappa_c at consolidated evidence', float(KC)],
        ['kappa_c at depleted evidence', float(KC_DEPLETED)],
        ['random seed', SEED],
        ['Fig. 1a', 'schematic; no underlying data'],
    ])
    T.update(SOURCE_DATA)

    rows = D.get('pulse_rows', [])
    if rows:
        T['Fig 1c'] = dict(
            columns=['pulse duration tau', 'amplitude', 'amplitude / critical',
                     'root at end of pulse', 'recovered'],
            rows=[[r['tau'], r.get('amplitude'), r.get('ratio'),
                   r['root_at_pulse_end'], bool(r['recovered'])] for r in rows])
    prop = D.get('propagation', {})
    if prop:
        T['Fig 1d'] = dict(columns=['kappa', 'r0 final', 'r1 final', 'r2 final', 'r3 final'],
                           rows=[[float(k)] + [float(x) for x in prop[k]]
                                 for k in sorted(prop, key=float)])
    noise = D.get('identification_noise', {})
    if noise:
        T['Fig 2c'] = dict(
            columns=['lognormal noise on the stage ratios', 'estimator',
                     'median kappa / true', 'IQR low', 'IQR high'],
            rows=[[float(pc), name,
                   noise[pc][key]['k_median'] / KAPPA_UNTREATED,
                   noise[pc][key]['k_iqr'][0] / KAPPA_UNTREATED,
                   noise[pc][key]['k_iqr'][1] / KAPPA_UNTREATED]
                  for pc in sorted(noise, key=float)
                  for key, name in (('primary', 'kappa = a r1 - r1^2'),
                                    ('two_ratio', 'two stage ratios'))])
    sw = D.get('stage2_sweep', [])
    if sw:
        T['Fig 3a'] = dict(columns=['kappa_2 (kappa_1 = kappa_3 = 0.2)',
                                    'r1 final', 'r2 final', 'r3 final'],
                           rows=[[s['kappa2']] + [float(x) for x in s['final'][1:]]
                                 for s in sw])
    ne = D.get('noise_escape', {})
    if ne:
        keys = sorted(ne, key=float)
        grid = ne[keys[0]]['survival_grid']
        T['Fig 3b'] = dict(
            columns=['time'] + ['fraction on the branch, sigma = %s' % k for k in keys],
            rows=[[grid[i]] + [ne[k]['survival'][i] for k in keys]
                  for i in range(len(grid))])
    mg = D.get('modulation_grid', [])
    if mg:
        T['Fig 4a'] = dict(columns=['duty fraction f', 'elevated coupling',
                                    'duty-weighted mean', 'escaped'],
                           rows=[[r['f'], r['kappa_hi'],
                                  r['f'] * r['kappa_hi'] + (1 - r['f']) * KAPPA_UNTREATED,
                                  bool(r['escaped'])] for r in mg])
    ps = D.get('period_sensitivity', [])
    if ps:
        T['Fig 4b'] = dict(
            columns=['period / static escape time', 'period', 'escape time',
                     'escape time / static', 'saturated in the first elevated phase'],
            rows=[[r['ratio'], r['period'], r['t'], r['ratio_to_static'],
                   bool(r['escaped_within_first_high_phase'])] for r in ps])
    sp = D.get('spectrum', [])
    if sp:
        T['Fig 3c'] = dict(
            columns=['threshold a', 'kappa / kappa_c', 'kappa', 'lambda_1', 'lambda_2',
                     'lambda_3', 'lambda_1 closed form'],
            rows=[[s['a'], s['frac'], s['kappa']] + list(s['eigenvalues'])
                  + [s['lambda1_closed_form']] for s in sp])
    et = D.get('escape_times', [])
    if et:
        T['Fig 4c'] = dict(columns=['excess (kappa - kappa_c)/kappa_c', 'escape time',
                                    'local slope'],
                           rows=[[r['excess'], r['t'], r['local_slope']] for r in et])
        fx, f0 = D.get('full_system_exponents', {}), D.get('full_system_exponents_eps0', {})
        if fx and f0:
            T['Fig 4c inset'] = dict(
                columns=['Q0', 'threshold a(Q0)', 'exponent below 5% excess, eps = 0.02',
                         'exponent below 5% excess, eps = 0'],
                rows=[[k.split('=')[1], fx[k]['a'], fx[k]['exponent_small_excess'],
                       f0[k]['exponent_small_excess']]
                      for k in sorted(fx, key=lambda s: fx[s]['a'])])
    tr = D.get('staircase', {}).get('trajectory')
    if tr:
        Tt = D['staircase']['T']
        T['Fig 4d'] = dict(columns=['time', 'time / escape time', 'r0', 'r1', 'r2', 'r3'],
                           rows=[[row[0], row[0] / Tt] + list(row[1:]) for row in tr])
    return T


def legend_panels():
    """The panels each figure legend declares, as {'Fig 1': ['a','b','c','d'], ...}.

    Source-data sheets and figure citations are checked against this, so a
    renumbering of the display items cannot silently leave either behind.
    """
    out = {}
    for kind, txt in MANUSCRIPT_PARAS:
        if kind != 'leg':
            continue
        m = re.match(r'\*\*Fig\. (\d+) \|', txt)
        if not m:
            continue
        out['Fig ' + m.group(1)] = re.findall(r'\*\*([a-z])\*\*,', txt)
    return out


def write_source_data(D, path='Source_Data_Figs1-4.xlsx'):
    """One worksheet per figure panel.  Values only, no formulas."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment
    except ImportError:
        print('openpyxl not installed; source data not written')
        return None
    T = source_data_tables(D)
    panels = legend_panels()
    stray = [n for n in T if n != 'Read me'
             and not (re.match(r'Fig (\d+)([a-z])', n)
                      and re.match(r'Fig (\d+)([a-z])', n).group(2)
                      in panels.get('Fig ' + re.match(r'Fig (\d+)', n).group(1), []))]
    if stray:
        raise ValueError('source-data sheets with no matching figure panel: %s'
                         % ', '.join(sorted(stray)))

    def key(name):                      # 'Read me' first, then panel order
        if name == 'Read me':
            return (0, 0, '')
        m = re.match(r'Fig (\d)([a-z])', name)
        return (1, int(m.group(1)), m.group(2) + name[len('Fig 1a'):]) if m else (2, 0, name)
    T = {k: T[k] for k in sorted(T, key=key)}
    wb = Workbook()
    wb.remove(wb.active)
    head = Font(name='Arial', size=10, bold=True)
    body = Font(name='Arial', size=10)
    for name, tab in T.items():
        ws = wb.create_sheet(name[:31])
        ws.append(tab['columns'])
        for c in ws[1]:
            c.font = head
            c.alignment = Alignment(horizontal='left')
        for row in tab['rows']:
            ws.append(row)
        for col in ws.iter_cols(min_row=2):
            for c in col:
                c.font = body
        for i, name_ in enumerate(tab['columns'], start=1):
            width = max(len(str(name_)) + 2, 12)
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, 42)
        ws.freeze_panes = 'A2'
    wb.save(path)
    print('wrote %s (%d panels)' % (path, len(T) - 1))
    return path


# --------------------------------------------------------------------------
# Verification: every number quoted in the manuscript, checked against results
# --------------------------------------------------------------------------
def verify(D):
    """Re-check each value the manuscript reports. Returns True if all pass.

    Values that depend on grid size are skipped when results.json came from a
    --quick run, since the manuscript reports the full grids.
    """
    quick = bool(D.get('parameters', {}).get('quick'))
    checks, skipped = [], 0

    def chk(label, got, want, tol=0.0, fmt='%.6g', grid=False):
        nonlocal skipped
        if grid and quick:
            skipped += 1
            return
        ok = (got == want) if tol == 0 and not isinstance(want, float) \
            else abs(float(got) - float(want)) <= tol
        checks.append((ok, label, fmt % got if not isinstance(got, str) else got,
                       fmt % want if not isinstance(want, str) else want))

    def shown(label, got, printed, grid=False):
        """The manuscript prints `printed`; check the computed value rounds to it.

        Targets stated at more digits than the text quotes can pass while the
        text is wrong, because rounding twice is not rounding once (4.56479
        rounds to 4.565 and then to 4.57, but to 4.56 in one step).  Comparing
        at the printed precision removes that failure mode.
        """
        d = len(printed.split('.')[1]) if '.' in printed else 0
        chk(label, round(float(got), d), float(printed), 0.0, '%.' + str(d) + 'f',
            grid=grid)

    chk('maximum principle: threshold crossings', D['max_principle_total_crossings'], 0)
    chk('maximum principle: graphs tested', D['max_principle_total_trials'], 400, grid=True)
    chk('maximum of r never rose (< 1e-12)',
        max(r['max_rise_of_maximum'] for r in D['max_principle']) < 1e-12, True)
    ce = D['reverse_coupling_counterexample']
    chk('reverse coupling alone carries the root across',
        any(c['root_crossed'] and c['kappa_forward'] == 0.0 for c in ce), True)
    chk('forward coupling alone does not', ce[0]['root_crossed'], False)
    chk('root trajectory independent of kappa (< 1e-8, dense comparison)',
        D['root_trajectory_deviation'] < 1e-8, True)

    chk('pulse criterion holds for every (amplitude, duration)',
        D['pulse_criterion_consistent'], True)
    chk('crossing threshold = a0', D['excursion_threshold'], A0, 1e-9)
    spread = max(D['excursion_threshold_vs_kappa'].values()) \
        - min(D['excursion_threshold_vs_kappa'].values())
    chk('crossing threshold independent of kappa (spread < 1e-9)', spread < 1e-9, True)
    chk('propagation boundary = a0^2/4', D['propagation_boundary'], KC_DEPLETED, 1e-6)
    shown('propagation boundary at the digits the text prints (0.0900005)',
          D['propagation_boundary'], '0.0900005')
    chk('K* of ref. 4 (sufficient, not sharp)', D['K_star_ref4'], 0.291, 5e-4, '%.4f')

    chk('critical coupling a1^2/4', D['kappa_c_closed_form'], 0.005625, 1e-9)
    tb = D['true_branch']
    for i, want in enumerate([1.0, 0.023765, 4.6763e-4, 9.1698e-6]):
        chk('exact partial branch r%d' % i, tb[i], want, abs(want) * 1e-3, '%.6g')
    comp = [c for c in D['branch_comparison'] if abs(c['kappa'] - KAPPA_UNTREATED) < 1e-12][0]
    chk('closed form overstates r2 by', comp['rel_error'][2], 0.0197, 5e-4, '%.4f')
    chk('closed form overstates r3 by', comp['rel_error'][3], 0.0400, 5e-4, '%.4f')
    chk('analytic residual matches the numeric one',
        max(abs(a - b) for a, b in zip(comp['analytic_residual'], comp['numeric_residual']))
        < 1e-15, True)
    chk('exact branch is an equilibrium (residual < 1e-12)',
        D['true_branch_residual'] < 1e-12, True)
    chk('branch exists at kappa = 0.005625', D['branch_exists']['0.005625'], True)
    chk('branch absent at kappa = 0.00563', D['branch_exists']['0.00563'], False)

    ident = [b for b in D['identification_bias'] if abs(b['kappa'] - KAPPA_UNTREATED) < 1e-12][0]
    chk('two-ratio inversion, bias in a on the exact branch',
        ident['a_rel_error'], -0.094, 2e-3, '%.3f')
    chk('two-ratio inversion, bias in kappa on the exact branch',
        ident['kappa_rel_error'], -0.112, 2e-3, '%.3f')
    chk('primary estimator is exact on the exact branch (< 1e-9)',
        abs(ident['primary_rel_error']) < 1e-9, True)

    panels = legend_panels()
    sheets = set(source_data_tables(D)) - {'Read me'}
    declared = set('%s%s' % (f, p) for f, ps in panels.items() for p in ps)
    stray = sorted(n for n in sheets if n.split(' inset')[0] not in declared)
    chk('every source-data sheet matches a figure panel', not stray, True)
    if SOURCE_DATA:      # populated by make_figures; empty when verifying alone
        missing = sorted(declared - sheets - {'Fig 1a'})
        chk('every data panel has a source-data sheet', not missing, True)

    mis = {('%.2f' % d['a_used']): d for d in D.get('threshold_misspecification', [])}
    if mis:
        chk('primary estimator with the prevailing threshold a(Q) = a1',
            mis['0.15']['ratio'], 1.000, 1e-6, '%.3f')
        shown('using a = 0.30 instead inflates kappa by (text: 2.19)',
              mis['0.30']['ratio'], '2.19')
        shown('using the depleted-evidence a0 instead inflates kappa by (text: 4.56)',
              mis['0.60']['ratio'], '4.56')
        shown('a 10% error in the prevailing threshold moves kappa by (text: 12%)',
              D['threshold_sensitivity_10pct'], '0.12')

    chk('escape exponent, local slope at the smallest excess',
        D['local_slope_at_smallest_excess'], 0.501, 2e-3, '%.4f', grid=True)
    chk('escape exponent, regression over excess <= 5%',
        D['exponent_small_excess'], 0.509, 1e-3, '%.3f', grid=True)
    chk('escape exponent, regression over the full range',
        D['exponent_full_range'], 0.529, 1e-3, '%.3f', grid=True)
    sc = D['staircase']
    chk('staircase crossing r1', sc['crossings']['r1'], 0.31, 0.01, '%.3f')
    chk('staircase crossing r2', sc['crossings']['r2'], 0.65, 0.01, '%.3f')
    chk('r1 transit from a1 to 0.9 (fraction of the interval)',
        sc['transit_a1_to_0p9']['r1'], 0.034, 3e-3, '%.3f')
    chk('r2 transit from a1 to 0.9 (fraction of the interval)',
        sc['transit_a1_to_0p9']['r2'], 0.033, 3e-3, '%.3f')
    st = sc['stasis_fraction_vs_threshold']
    chk('stasis fraction is threshold dependent (0.5% vs 20% of max rate)',
        float(st['0.2']) - float(st['0.005']) > 0.3, True)

    chk('duty-weighted criterion agreement', D['modulation_agreement'], 40, grid=True)
    chk('modulation cells', D['modulation_n'], 42, grid=True)
    fails = D['modulation_failures']
    chk('both criterion failures at the lowest duty fraction',
        all(g['f'] == min(x['f'] for x in D['modulation_grid']) for g in fails), True)
    P = D['period_sensitivity']
    long_p = [p for p in P if p['ratio'] >= 1.0 and p['ratio_to_static']]
    chk('at long period the modulated system escapes sooner than the average predicts',
        all(p['ratio_to_static'] < 1 for p in long_p), True, grid=True)
    chk('long-period points are saturated inside the first elevated phase',
        all(p['escaped_within_first_high_phase'] for p in long_p), True, grid=True)

    fx = D.get('full_system_exponents', {})
    if fx:
        vals = [v['exponent_full_range'] for v in fx.values()]
        sm = [v['exponent_small_excess'] for v in fx.values() if v['exponent_small_excess']]
        chk('full-system exponents (full range), minimum', min(vals), 0.478, 5e-3, '%.3f', grid=True)
        chk('full-system exponents (full range), maximum', max(vals), 0.570, 5e-3, '%.3f', grid=True)
        chk('full-system small-excess exponents reported', len(sm) == len(fx), True, grid=True)

    loc = D.get('annihilation_locus', {})
    for kappa, want in (('0.05', 0.5827), ('0.08', 0.2761), ('0.02', 0.8395)):
        if kappa in loc:
            chk('annihilation locus Q* at kappa = %s' % kappa,
                loc[kappa]['Q_star'], want, 1e-4, '%.4f')
    if '0.006' in loc:
        chk('Q* tends to 1 at the lower edge of the band',
            loc['0.006']['Q_star'] > 0.99, True)
    # Each lag is checked against the value the manuscript prints, not against a
    # blanket tolerance: the three measured lags differ by a factor of sixty and
    # a single loose bound would hide that.
    for key, want in (('s=0.01,kappa=0.05', 0.0029), ('s=0.01,kappa=0.08', 0.1130),
                      ('s=1,kappa=0.08', 0.0017)):
        v = D.get('delayed_completion', {}).get(key)
        if v and v.get('t') and v.get('t_annihilation'):
            lag = 100 * (v['t'] - v['t_annihilation']) / v['t_annihilation']
            chk('escape follows the fold by %% of annihilation time (%s)' % key,
                lag, want, max(5e-4, 0.05 * want), '%.4f')
    for key in ('s=1,kappa=0.05', 's=0.01,kappa=0.02', 's=1,kappa=0.02',
                's=0.01,kappa=0.003', 's=1,kappa=0.003'):
        v = D.get('delayed_completion', {}).get(key)
        if v is not None:
            chk('no escape within the integration horizon (%s)' % key,
                bool(v['escaped']), False)

    # eps = 0 control: with the evidence channel off the exponent must no longer
    # depend on the threshold, and it must be the one the frozen fold gives.
    f0 = D.get('full_system_exponents_eps0', {})
    if f0:
        fr = [v['exponent_full_range'] for v in f0.values()]
        sm0 = [v['exponent_small_excess'] for v in f0.values()
               if v['exponent_small_excess']]
        chk('control (eps=0): full-range exponents collapse, minimum',
            min(fr), 0.570, 5e-3, '%.4f', grid=True)
        chk('control (eps=0): full-range exponents collapse, maximum',
            max(fr), 0.577, 5e-3, '%.4f', grid=True)
        chk('control (eps=0): small-excess exponents collapse, minimum',
            min(sm0), 0.527, 5e-3, '%.4f', grid=True)
        chk('control (eps=0): small-excess exponents collapse, maximum',
            max(sm0), 0.532, 5e-3, '%.4f', grid=True)
        chk('control (eps=0): spread across thresholds is below 0.01',
            max(sm0) - min(sm0) < 0.01, True, grid=True)
        if fx:
            sm2 = [v['exponent_small_excess'] for v in fx.values()
                   if v['exponent_small_excess']]
            chk('evidence feedback lowers the small-excess exponent to',
                min(sm2), 0.386, 5e-3, '%.4f', grid=True)
            chk('the depression is absent from the control',
                (max(sm0) - min(sm0)) < 0.1 * (max(sm2) - min(sm2)), True, grid=True)
    deep = D.get('eps0_deep_small_excess', {})
    for key, v in deep.items():
        chk('control (eps=0) local slope at excess 2e-4 (%s)' % key,
            v['local_slopes'][0], 0.503, 3e-3, '%.4f', grid=True)
    sf = D.get('start_fraction_of_remnant', {})
    if sf:
        chk('the start sits at a fixed fraction of the branch remnant',
            max(abs(x - 0.2929) for x in sf.values()) < 1e-3, True)
    # Every number printed in the Supplementary table and in the deep-control
    # sentence is re-read out of the document text and compared with the value
    # the analysis produced, so a transcription slip cannot survive a run.
    text = ' '.join(v for _, v in SUPPLEMENT_PARAS)
    rows = re.findall(r'Q\u2080 = ([\d.]+);\s+a = ([\d.]+);\s+\u03b5 = 0\.02: ([\d.]+), '
                      r'([\d.]+);\s+\u03b5 = 0: ([\d.]+), ([\d.]+)', text)
    if rows and fx and f0:
        wrong = []
        for r in rows:
            key = 'Q0=%g' % float(r[0])
            if key not in fx or key not in f0:
                continue        # a --quick run sweeps only three evidence levels
            printed = [float(x) for x in r[1:]]
            computed = [fx[key]['a'], fx[key]['exponent_full_range'],
                        fx[key]['exponent_small_excess'],
                        f0[key]['exponent_full_range'], f0[key]['exponent_small_excess']]
            for got, want, dp in zip(printed, computed, (4, 3, 3, 3, 3)):
                if round(want, dp) != got:
                    wrong.append('%s %.*f vs %.*f' % (key, dp, got, dp, want))
        chk('Supplementary table matches the computed exponents',
            'none' if not wrong else '; '.join(wrong), 'none', grid=True)
        chk('Supplementary table covers every evidence level',
            len(rows), len(fx), grid=True)
    sent = [v for _, v in SUPPLEMENT_PARAS
            if 'local slope between consecutive excesses' in v]
    if sent and deep:
        printed = [float(x) for x in re.findall(r'0\.\d{3}(?!\d)', sent[0])]
        computed = [round(x, 3) for k in ('Q0=0.5', 'Q0=0')
                    for x in deep[k]['local_slopes'][::-1]]
        chk('Supplementary deep-control slopes match', printed, computed, grid=True,
            fmt='%s')

    het = D.get('heterogeneous_thresholds')
    if het:
        chk('maximum principle with unequal thresholds: crossings of min a',
            het['crossings'], 0)
        chk('unequal-threshold graphs tested', het['trials'], 200, grid=True)
        chk('maximum never rose there either (< 1e-12)',
            het['max_rise_of_maximum'] < 1e-12, True)

    sb = D.get('stage_boundaries', {})
    for key, got in sb.items():
        chk('per-stage propagation boundary = a0^2/4 (%s)' % key, got, KC_DEPLETED,
            1e-6, '%.7f')
    ar = D.get('arrest_depth', [])
    if ar:
        chk('arrest depth = index of the first subcritical stage',
            all(r['depth'] == r['predicted_depth'] for r in ar), True)
        chk('a chain with every coupling above a0^2/4 recovers completely',
            [r['depth'] for r in ar if min(r['kappa']) > KC_DEPLETED], [3] *
            len([r for r in ar if min(r['kappa']) > KC_DEPLETED]), fmt='%s')
    ex = D.get('branch_exists_vs_kappa1', {})
    if ex:
        chk('branch existence depends on kappa_1 alone',
            all(v for k, v in ex.items() if k.startswith('k1=0.005625'))
            and not any(v for k, v in ex.items() if k.startswith('k1=0.00563')), True)
    L = D.get('kappa_c_vs_chain_length', {})
    if L:
        chk('kappa_c is the same at every chain length (spread < 1e-11)',
            max(L.values()) - min(L.values()) < 1e-11, True)
        chk('kappa_c at every chain length', min(L.values()), 0.005625, 1e-11, '%.12f')

    ac = D.get('additive_coupling', {})
    if ac:
        for kind, v in ac.items():
            chk('additive coupling converts a sub-threshold chain (%s)' % kind,
                any(r['crossed'] for r in v['rows']), True)
            chk('additive coupling: critical weight (%s)' % kind, v['critical_w'],
                {'linear': 0.100, 'sigmoid': 0.060}[kind], 5e-3, '%.3f')
        chk('diffusive coupling never does, up to w = 100',
            any(d['crossed'] for d in D['diffusive_control']), False)
    ti = D.get('threshold_intervention', [])
    if ti:
        chk('lowering the threshold alone recovers the chain',
            [d['a'] for d in ti if d['recovered']], [0.25], fmt='%s')
    pr = D.get('root_above_threshold', [])
    if pr:
        chk('a root above threshold converts the chain exactly when kappa > a0^2/4',
            all(d['chain_recovered'] == d['supercritical'] for d in pr), True)

    sw = D.get('stagewise_band', [])
    if sw:
        chk('stage-wise band: every completion sits above the running maximum of Q*',
            all(r['overshoot'] >= 0 for v in sw for r in v['stages']
                if r['overshoot'] is not None), True)
        chk('stage-wise band: largest overshoot',
            D['stagewise_overshoot_max'], 0.041, 1e-3, '%.4f', grid=True)
        chk('stage-wise band: completions follow chain order',
            all([r['t'] for r in v['stages'] if r['t'] is not None] ==
                sorted(r['t'] for r in v['stages'] if r['t'] is not None) for v in sw), True)
        chk('stage-wise band: a stage below the running maximum is gated, not free',
            all(r['gated_by_predecessor'] == (r['Q_star_own'] < r['Q_star_running_max'] - 1e-12)
                for v in sw for r in v['stages']), True)
    nsc = D.get('noise_step_control')
    if nsc:
        ne8 = D.get('noise_escape', {}).get('0.008', {})
        chk('halving the noise step leaves the escaped fraction alone (within 5%)',
            abs(nsc['escaped'] - ne8.get('escaped', 0)) <= 0.05 * nsc['paths'], True,
            grid=True)
        chk('but moves the median, so only its order is quoted',
            0.5 < nsc['median'] / ne8['median'] < 2.0, True, grid=True)

    lem = D.get('induction_lemma', {})
    for key, v in lem.items():
        chk('comparison lemma: field bounded below on [0,0.9] (kappa=%s)' % key,
            v['field_min_on_0_theta'] >= v['analytic_lower_bound'] > 0, True)
        chk('comparison lemma: stage crosses within the bound (kappa=%s)' % key,
            v['crossing_time'] is not None and
            v['crossing_time'] < v['crossing_time_bound'], True)

    sp = D.get('spectrum')
    if sp:
        chk('lambda_1 equals -sqrt(a^2-4k)(1-r1) (deviation < 1e-14)',
            D['spectrum_max_deviation'] < 1e-14, True)
        chk('every eigenvalue on the branch is negative',
            D['spectrum_largest_eigenvalue'] < 0, True)
        chk('lambda_1 is also the least negative on the tested grid',
            D.get('spectrum_lambda1_is_least_negative'), True, grid=True)
        # the rate must vanish AS A SQUARE ROOT: a hundredfold approach to the
        # fold has to divide the slowest rate by ten, and does, to 0.3%
        ratios = []
        for a in {s['a'] for s in sp}:
            far = [s for s in sp if s['a'] == a and abs(s['frac'] - 0.999) < 1e-9]
            near = [s for s in sp if s['a'] == a and abs(s['frac'] - 0.99999) < 1e-9]
            if far and near:
                ratios.append(abs(far[0]['eigenvalues'][0] / near[0]['eigenvalues'][0]))
        if ratios:
            chk('slowest rate falls by 10 for a 100-fold approach to the fold',
                min(ratios), 10.0, 0.05, '%.3f', grid=True)

    wd = D.get('well_depth', {}).get('0.003')
    if wd:
        chk('barrier of the confined well at kappa = 0.003', wd['barrier'], 1.66e-4,
            5e-6, '%.3e')
        chk('the unstable root it must clear', wd['r_unstable'], 0.126235, 1e-5, '%.6f')
    ne = D.get('noise_escape', {})
    if ne.get('0.008') and not ne['0.008'].get('quick'):
        v = ne['0.008']
        chk('noise sigma = 0.008 empties the confined well', v['escaped'] > 0.9 * v['paths'],
            True, grid=True)
        chk('median noise escape time is of order 1e3 to 1e4',
            1e3 < (v['median'] or 0) < 1e4, True, grid=True)
    if ne.get('0.002'):
        chk('noise sigma = 0.002 does not', ne['0.002']['escaped'], 0, grid=True)

    ad = D.get('arrest_drift', {}).get('0.003')
    if ad:
        chk('confined branch r1 at depleted evidence', ad['profile_at_a0'][1], 0.005042,
            1e-5, '%.5f')
        chk('confined branch r1 at consolidated evidence', ad['profile_at_a1'][1], 0.023765,
            1e-5, '%.5f')

    width = max(len(c[1]) for c in checks)
    print('\n%-*s   %12s   %12s' % (width, 'claim', 'computed', 'reported'))
    print('-' * (width + 32))
    for ok, label, got, want in checks:
        print('%-*s   %12s   %12s   %s' % (width, label, got, want, 'ok' if ok else 'FAIL'))
    bad = [c for c in checks if not c[0]]
    print('-' * (width + 32))
    print('%d checks, %d failed%s' % (len(checks), len(bad),
          ' (%d grid-size-dependent checks skipped: results.json is from a '
          '--quick run)' % skipped if skipped else ''))
    return not bad


# --------------------------------------------------------------------------
def run_all(quick=False):
    rng = np.random.default_rng(SEED)
    out = {'parameters': dict(a0=A0, a1=A1, p=P_EXP, eps=EPS, rho=RHO,
                              kappa_illustrative=KAPPA_REF,
                              kappa_untreated=KAPPA_UNTREATED,
                              K_star=float(K_STAR), seed=SEED, quick=quick)}
    print('[1/11] confinement below threshold on arbitrary graphs ...', flush=True)
    out.update(analysis_max_principle(rng, quick))
    print('      %d of %d runs crossed the threshold; largest rise of the maximum %.1e'
          % (out['max_principle_total_crossings'], out['max_principle_total_trials'],
             max(r['max_rise_of_maximum'] for r in out['max_principle'])))
    ce = [c for c in out['reverse_coupling_counterexample'] if c['root_crossed']]
    print('      reverse coupling carries the root across in %d of %d counterexample cells'
          % (len(ce), len(out['reverse_coupling_counterexample'])))

    print('[2/11] root pulse and the crossing criterion ...', flush=True)
    out.update(analysis_pulse(quick))
    print('      pulse criterion r0(tau) > a0 held in every (amplitude, duration) cell: %s'
          % out['pulse_criterion_consistent'])
    print('      crossing threshold %.10f' % out['excursion_threshold'])

    print('[3/11] partial branch, critical coupling, identification ...', flush=True)
    out.update(analysis_branch(rng, quick))
    comp = [c for c in out['branch_comparison'] if abs(c['kappa'] - KAPPA_UNTREATED) < 1e-12][0]
    print('      closed form overstates r2 by %.2f%% and r3 by %.2f%%'
          % (100 * comp['rel_error'][2], 100 * comp['rel_error'][3]))
    ident = [b for b in out['identification_bias'] if abs(b['kappa'] - KAPPA_UNTREATED) < 1e-12][0]
    mm = {('%.2f' % d['a_used']): d['ratio'] for d in out['threshold_misspecification']}
    print('      threshold misspecification: a=0.30 inflates kappa x%.2f, a0=0.60 x%.2f'
          % (mm['0.30'], mm['0.60']))
    print('      two-ratio inversion on the exact branch: a %+.1f%%, kappa %+.1f%%; '
          'primary estimator %+.1e%%'
          % (100 * ident['a_rel_error'], 100 * ident['kappa_rel_error'],
             100 * ident['primary_rel_error']))
    print('      propagation boundary %.8f (a0^2/4 = %.8f)'
          % (out['propagation_boundary'], KC_DEPLETED))

    print('[4/11] escape-time scaling and staircase ...', flush=True)
    out.update(analysis_escape(quick))
    print('      local slope %.4f at the smallest excess; regression over excess<=5%%: %.4f'
          % (out['local_slope_at_smallest_excess'], out['exponent_small_excess']))

    print('[5/11] intermittent modulation and period sensitivity ...', flush=True)
    out.update(analysis_modulation(quick))
    print('      criterion agrees in %d of %d cells'
          % (out['modulation_agreement'], out['modulation_n']))

    print('[6/11] full system: delayed completion and confirmatory exponent ...', flush=True)
    out.update(analysis_full(quick))
    for key, v in out['delayed_completion'].items():
        if v.get('t') and v.get('t_annihilation'):
            print('      %s: fold at t=%.3e, escape at t=%.3e (%.3f%% later)'
                  % (key, v['t_annihilation'], v['t'],
                     100 * (v['t'] - v['t_annihilation']) / v['t_annihilation']))
    fx = out['full_system_exponents']
    if fx:
        vals = [v['exponent_full_range'] for v in fx.values()]
        sm = [v['exponent_small_excess'] for v in fx.values() if v['exponent_small_excess']]
        print('      full-system exponent %.2f to %.2f (full range), %.2f to %.2f '
              '(excess <= 5%%), across %d evidence levels'
              % (min(vals), max(vals), min(sm), max(sm), len(fx)))
    a = out['arrest_drift']['0.003']
    print('      confined branch drift at kappa=0.003: r1 %.5f -> %.5f (x%.2f)'
          % (a['profile_at_a0'][1], a['profile_at_a1'][1], a['first_stage_gain']))

    print('[7/11] unequal coupling: the weakest stage ...', flush=True)
    out.update(analysis_unequal(quick))
    sb = out['stage_boundaries']
    print('      per-stage boundaries %s (a0^2/4 = %.9f)'
          % (', '.join('%.9f' % v for v in sb.values()), KC_DEPLETED))
    ok = all(r['depth'] == r['predicted_depth'] for r in out['arrest_depth'])
    print('      arrest depth equals the first subcritical stage in every case: %s' % ok)
    L = out['kappa_c_vs_chain_length']
    print('      kappa_c over chain lengths 1 to %d: %.12f to %.12f'
          % (len(L), min(L.values()), max(L.values())))

    print('[8/11] spectrum on the branch ...', flush=True)
    out.update(analysis_spectrum(quick))
    print('      |lambda_1 - closed form| <= %.1e; largest eigenvalue %.3e'
          % (out['spectrum_max_deviation'], out['spectrum_largest_eigenvalue']))

    print('[9/11] well depth and noise ...', flush=True)
    out.update(analysis_noise(quick))
    w = out['well_depth']['0.003']
    print('      barrier at kappa=0.003: %.2e' % w['barrier'])
    for k, v in out['noise_escape'].items():
        print('      sigma=%s: %d of %d paths left the well within %.0e%s'
              % (k, v['escaped'], v['paths'], v['horizon'],
                 '' if v['median'] is None else ', median %.0f' % v['median']))
    c = out.get('noise_step_control')
    if c:
        print('      step control (dt halved): %d of %d escaped, median %.0f'
              % (c['escaped'], c['paths'], c['median']))

    print('[10/11] the intermediate band, stage by stage ...', flush=True)
    out.update(analysis_stagewise_band(quick))
    for v in out['stagewise_band']:
        done = [r for r in v['stages'] if r['Q'] is not None]
        print('      kappa=%s: %s'
              % (v['kappa'], ', '.join('stage %d at Q=%.4f (predicted %.4f)'
                                       % (r['stage'], r['Q'], r['Q_star_running_max'])
                                       for r in done)))
    print('      overshoot above the running maximum of Q*: %+.4f to %+.4f over %d events'
          % (out['stagewise_overshoot_min'], out['stagewise_overshoot_max'],
             out['stagewise_events']))

    print('[11/11] what the confinement theorem does not cover ...', flush=True)
    out.update(analysis_coupling_form(quick))
    for kind, v in out['additive_coupling'].items():
        print('      additive (%s): converts a sub-threshold chain above w = %.3f'
              % (kind, v['critical_w']))
    print('      diffusive control: no crossing up to w = %g'
          % max(d['w'] for d in out['diffusive_control']))
    t = [d for d in out['threshold_intervention'] if d['recovered']]
    print('      lowering the threshold alone recovers the chain at a = %s'
          % ', '.join('%.2f' % d['a'] for d in t))
    pr = out['root_above_threshold']
    print('      root left at 0.61: chain recovers for kappa = %s, not for %s'
          % (', '.join('%g' % d['kappa'] for d in pr if d['chain_recovered']),
             ', '.join('%g' % d['kappa'] for d in pr if not d['chain_recovered'])))
    return out


def in_notebook():
    """True inside Jupyter, Colab or any other IPython kernel."""
    return 'ipykernel' in sys.modules or 'google.colab' in sys.modules


def run(quick=False, figures=False, verify=False, docx=None, outdir=None):
    """Entry point for notebooks.

    run()                      full analysis, then the figures
    run(quick=True)            the same on coarser grids, a few minutes
    run(verify=True)           re-check every reported value
    run(docx='template.docx')  write the manuscript and the Supplementary file
    Flags compose, and anything a later step needs is produced first.
    """
    argv = []
    if quick:
        argv.append('--quick')
    if figures:
        argv.append('--figures')
    if verify:
        argv.append('--verify')
    if docx:
        argv += ['--docx', docx]
    if outdir:
        argv += ['--outdir', outdir]
    return main(argv)


def show_figures():
    """Display Fig1-4 inline when running in a notebook; a no-op otherwise."""
    if not in_notebook():
        return
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    for name in ('Fig1.png', 'Fig2.png', 'Fig3.png', 'Fig4.png'):
        if os.path.exists(name):
            display(Image(filename=name))




# ==========================================================================
# Manuscript and Supplementary Information
# ==========================================================================
# The paragraph lists below are the text of the two documents.  `build_docx`
# renders them into Word files, reusing the page setup and styles of a template
# .docx supplied with --docx; only the template's section properties, styles and
# footers are used, never its text.
#
#   kind:   'title' | 'h1' (bold heading) | 'h2' (plain subheading) | 'p'
#           'eq' (centred) | 'leg' (single-spaced figure legend)
#           'fig' (image file) | 'pagebreak' | 'blank' | 'tab'
#   markup: ^{superscript}  _{subscript}  **bold**  *italic*

MANUSCRIPT_TITLE = 'Catatonia recovery separates two intervention axes that constrain electroconvulsive therapy'

MANUSCRIPT_PARAS = [
 ('title', MANUSCRIPT_TITLE),
 ('blank', ''),
 ('p', 'Hiroki Saito^{1,2*}, M.D., Ph.D.'),
 ('p', '^{1}Onda-daini Hospital, Matsudo-shi, Chiba, Japan'),
 ('p', '^{2}Department of Neuropsychiatry, Graduate School of Medicine, Nippon Medical School, Sendagi 1-1-5, Bunkyo-ku, Tokyo 113-8602, Japan'),
 ('blank', ''),
 ('tab', ''),
 ('p', '* Corresponding author: Hiroki Saito'),
 ('p', 'Department of Neuropsychiatry, Graduate School of Medicine, Nippon Medical School, Sendagi 1-1-5, Bunkyo-ku, Tokyo 113-8602, Japan'),
 ('p', 'E-mail: hiroki.saito.neuro@gmail.com'),
 ('p', 'Phone: +81-3-3822-2131; Fax: +81-3-5814-6287'),
 ('p', 'ORCID: 0000-0002-4964-8559'),
 ('blank', ''),
 ('blank', ''),
 ('h1', 'Abstract'),
 ('p', 'Catatonia recovers through capacities returning in a fixed dependency order, and benzodiazepines and electroconvulsive therapy work without a known target. On an evidence-coupled bistable cascade over that order we take three quantities to be externally accessible: the root capacity, the coupling between capacities, and accumulated evidence. Here we show that evidence has no external port and the other two act on disjoint regions. While every capacity is below threshold, a maximum principle forbids non-negative diffusive coupling, on any graph and either direction, from crossing the switching surface, so acute conversion needs an excursion, whereas an additive excitatory drive converts on its own. Above one exact criterion, the prevailing threshold squared over four, propagation separates into immediate, metastable and confined. On the modelled branch coupling alone shifts the resting first stage, so a treatment effective in both states is predicted to have separable excursion and coupling components, identifiable in model coordinates.'),
 ('blank', ''),
 ('h1', 'Introduction'),
 ('p', 'Recovery from catatonia proceeds through five precision domains in a fixed order, which held without inversion in twenty-five consecutive inpatients managed without electroconvulsive therapy (ECT)^{1}. That order is the unique linear extension of a dependency chain derived from the structural identifiability of an active-inference generative model^{2}, and for any dependency graph the admissible orders are its linear extensions^{3}. Placing an evidence-coupled bistable cascade on such a chain turns the structure into a recovery geometry with two stable configurations, an exactly invariant switching surface and, at weak coupling, a stable partial state^{4}. The four faster domains enter as capacities and the slowest, the precision of the slow environmental model, as the evidence coordinate rather than a fifth capacity^{4}. Existing accounts of catatonia are cast in terms of top-down modulation and network dysfunction^{5,6}; this one is about recovery order, and asks what an intervention can do.'),
 ('p', 'The question is sharply constrained once the intervention architecture is fixed. We take three quantities to be externally accessible: the capacity state, through an excursion applied to the root; the coupling between adjacent capacities; and the evidence coordinate. That choice is a modelling assumption, not a theorem, since the threshold parameters could be moved as well, and we return to it below. An interventional theory is then a statement about which of the three matters, when, and how the answer could be checked.'),
 ('p', 'Catatonia supplies the case in which the question is not academic. Benzodiazepines and ECT are the cornerstones of management, ECT the established escalation when benzodiazepines fail^{7}. Reported response to ECT lies between 59% and 100% across series, is preserved in benzodiazepine non-responders, and often follows one or a few treatments^{8-11}. Chronic partial states respond poorly^{9}. Efficacy is established; the target is not.'),
 ('p', 'The coupling axis is the tempting one, and the geometry settles what it can do. While every capacity is below threshold, a maximum principle forbids any non-negative diffusive coupling from producing a crossing, whatever the graph and in either direction, and that is the configuration in which acute treatment is most decisive. The word diffusive carries the argument: coupling entering as an additive excitatory drive rather than as a difference is not covered, and converts a wholly sub-threshold chain on its own. We keep κ as a phenomenological effective coupling defined by the dynamics, its biological implementation unspecified.'),
 ('p', 'In this work we separate the two remaining axes instead of choosing between them. The confinement result needs no chain structure; the results after the crossing use the forward chain, an assumption about the dynamics rather than a restatement of the dependency order. We show that the crossing is set by the root excursion, that propagation afterwards is set by coupling through one exact criterion, and that the two carry opposite temporal signatures. We then give an inversion that recovers the coupling from the first stage of the resting profile, which is what would make the division of labour testable.'),
 ('blank', ''),
 ('h1', 'Results'),
 ('h2', 'A recovery geometry with three external targets'),
 ('p', 'Four capacities r₀ to r₃ in [0,1] lie along a forward acyclic dependency chain, r₀ the root, and Q in [0,1] is a monotone coordinate on the accumulated Fisher information of a slow environmental model, attaining 1 only as a compactified limit^{4}. With z = r₀ − a(Q),'),
 ('eq', '*dr₀/dt = g(r₀,Q) + u(t),    drₖ/dt = g(rₖ,Q) + κ (rₖ₋₁ − rₖ),  k = 1,2,3,*'),
 ('eq', '*dQ/dt = ε [ χ₊(z) C(r) (1 − Q) − ρ χ₋(z) Q ],*'),
 ('p', 'with g(r,Q) = r(1 − r)(r − a(Q)), C(r) = ∏ⱼ rⱼ and a(Q) = a₀ − (a₀ − a₁)Qᵖ, p > 1; the selectors χ₊ and χ₋ are non-negative, vanish on z ≤ 0 and z ≥ 0 respectively, and are locally Lipschitz. Four properties used below are proved in Supplementary Note 1: stability of the recovered configuration for every κ ≥ 0 and of the failed configuration for ρ > 0, exact invariance of S = {r₀ = a(Q)} at a position independent of the evidence timescale, forward invariance of z > 0 and z < 0, and a bound K* above which every trajectory off S reaches a corner. Numerical values use the unfitted illustrative set of ref. 4, a₀ = 0.60, a₁ = 0.15 and κ = 0.6. That the coupling runs only forward is a further assumption, on which every result after the crossing rests and the result before it does not.'),
 ('p', 'Three quantities carry an external port in this architecture: u, κ and Q (Fig. 1a). Throughout, κ is an effective dynamical coupling between coarse-grained capacities, entering as a difference κ(rₖ₋₁ − rₖ), and not anatomical or statistical connectivity between brain regions. Which quantities are accessible is a modelling choice, not a derivation: a₀, a₁, p, ε and ρ are parameters too, and lowering a alone carries a sub-threshold chain to recovery with no input and no change of coupling, at a = 0.25 from a start at 0.30 (Supplementary Note 6). We fix u and κ as the intervention coordinates and treat a(Q) as switching geometry rather than a control.'),
 ('h2', 'Evidence has no external port'),
 ('p', 'That the evidence coordinate has no input port is a modelling choice, not a derivation. Its only inflow is the joint availability C(r), and identifying Q with the posterior precision of the slow model makes the choice coherent^{4,12}: Q is the identifiability accumulated over contexts in which every capacity was simultaneously online, and contexts in which any was offline contribute nothing. An additive exogenous source would break that. What is closed is the port, not the quantity: Q can still be filled sooner indirectly, since anything bringing capacities online earlier raises C(r). A reading admitting externally supplied context would place a port there; we do not, so every route to consolidation runs through the capacity state.'),
 ('h2', 'Confinement below threshold is a maximum principle'),
 ('p', 'The strongest statement here does not use the chain at all. Consider the same bistable law on an arbitrary graph with diffusive coupling, drₖ/dt = rₖ(1 − rₖ)(rₖ − a) + Σⱼ wₖⱼ (rⱼ − rₖ) with every weight wₖⱼ ≥ 0, and let M(t) = maxₖ rₖ(t). At an index attaining the maximum every difference rⱼ − rₖ is non-positive, so the coupling term is non-positive there and the upper Dini derivative obeys'),
 ('eq', '*D⁺M ≤ M(1 − M)(M − a) < 0    whenever  0 < M < a.*'),
 ('p', 'The set {maxₖ rₖ < a} is therefore forward invariant for every topology, every non-negative weight and both directions of coupling, and while 0 < M < a the maximum strictly decreases. The argument is the standard positively-invariant-region estimate for cooperative systems^{13,14} applied to the sup-norm: redistribution among capacities all below threshold cannot lift any across it, because it cannot raise the largest one. Integrating 400 systems from four graph families, feed-forward chains, symmetric graphs, purely reverse chains and general directed graphs, with about a third of the admissible weights at zero and the rest scaled by 0.5, 3 or 30, no trajectory started below a₀ reached the threshold and the maximum never rose (Fig. 1b). Thresholds need not be equal: the invariant set weakens to {maxₖ rₖ < minₖ aₖ}, as 200 graphs with a threshold per node confirmed.'),
 ('p', 'The scope of the statement is its hypothesis, and two counterexamples fix it. The first concerns the configuration: from r = (0.30, 0.90, 0.90, 0.90) the maximum is not below a, and a reverse coupling of 0.10 then converts the root. The second, and the more important, concerns the form of the coupling. The sign argument uses the difference rⱼ − rₖ, so it does not cover coupling entering additively, as an increase in cell-assembly drive would^{15}. Under a non-negative excitatory drive w(1 − rₖ)Σⱼ S(rⱼ), which keeps the state in [0,1], a chain started at 0.30 everywhere with no external input converts completely above w = 0.100 for S(r) = r and above w = 0.060 for a sigmoidal S, whereas the diffusive form never converts it at weights up to 100 (Supplementary Note 6). What is forbidden is diffusive crossing out of a configuration in which nothing is above threshold, not coupling-driven crossing in general.'),
 ('p', 'In the forward chain the conclusion is sharper: κ multiplies zero in the root equation, so the root trajectory is exactly independent of κ, confirmed to solver tolerance at κ = 0.6, 3.0 and 30.0. The premise the argument needs is not symmetry, and not a chain, but only that nothing is yet above threshold.'),
 ('p', 'Two literatures bound the claim. Confinement is a positively-invariant-region argument of the kind standard for cooperative systems^{13,14}, and pinning of fronts by discreteness on graphs is known^{16}. Propagation failure in discrete bistable chains and its critical coupling have been studied in detail since Keener^{17,18,19}. Our criterion is not that one: those results concern a front free to move on both sides, whereas κ_{c} = a(Q)²/4 is the discriminant of the clamped problem with the root held at 1, which is what an intervention at the root creates, and it is exact rather than variational. What we take from that literature is that the number depends on the coupling form, which is why we state the form each time.'),
 ('h2', 'The root excursion sets an exact and sharp crossing'),
 ('p', 'External input enters the root alone. The root field has a single interior unstable zero at r₀ = a(Q) and no incoming coupling, which makes its criterion exact rather than approximate. Under a rectangular pulse from the collapsed chain, swept over five durations and four amplitudes at the reference coupling, the outcome was fixed in every cell by the root value at the end of the pulse: the chain converted if and only if r₀(τ) > a₀ (Fig. 1c). Amplitude and duration are not separate levers, mattering only through where they leave the root. That the located boundary is a₀ to twelve digits is a check on the solver, the root being a one-dimensional bistable scalar. The equivalence is between the pulse and the crossing of the root, and the reference coupling is supercritical: below a₀²/4 the root recovers and the chain does not follow (Supplementary Note 6). Equivalently, an excursion from full recovery is absorbed when Δ < Δ_{c}(Q) = 1 − a(Q).'),
 ('h2', 'A closed-form criterion for the coupling axis'),
 ('p', 'After a crossing the root sits at its maximum and the profile is maximally unequal, so the coupling term dominates downstream. With r₀ = 1 the first-stage equilibrium condition factorises, and this is the one stage at which it does,'),
 ('eq', '*r₁(1 − r₁)(r₁ − a) + κ(1 − r₁) = (1 − r₁)[r₁² − a r₁ + κ] = 0,*'),
 ('p', 'so the low first-stage roots are r₁ = [a ± √(a² − 4κ)]/2 and exist only while a² − 4κ ≥ 0. The stable and unstable partners collide at a saddle-node bifurcation^{20} when the discriminant vanishes, giving an exact critical coupling'),
 ('eq', '*κ_{c}(Q) = a(Q)² / 4.*'),
 ('p', 'The factorisation is available only because r₀ = 1. Downstream the condition is the cubic rₖ(1 − rₖ)(rₖ − a) + κ(rₖ₋₁ − rₖ) = 0, and repeating the first-stage construction with κ rₖ₋₁ in place of κ leaves the residual κ rₖ(rₖ₋₁ − 1), which vanishes only at rₖ₋₁ = 1. The stage-wise closed form is therefore an approximation from the second stage on, overstating r₂ by 2.0% and r₃ by 4.0% at κ = 0.003 (Fig. 2a, inset). Because 0 < rₖ₋₁ < a the cubic changes sign on (0,a), so the exact low root is bracketed and obtained to machine precision, and every branch quantity uses it; the first stage and the critical coupling are unaffected.'),
 ('p', 'At consolidated evidence κ_{c} = a₁²/4 = 0.0056, and the branch exists at κ = 0.005625 and is absent at 0.00563 (Fig. 2a). At depleted evidence κ_{c} = a₀²/4 = 0.090, and the coupling at which a crossing propagates to the distal capacity was located at 0.0900005 (Fig. 1d). The criterion is sharp, whereas the bound K* ≈ 0.291 of ref. 4 is sufficient for convergence from every initial condition and a factor of three conservative. And κ_{c} moves with the threshold, falling by (a₀/a₁)² = 16 between a depleted and a consolidated system.'),
 ('p', 'Equal coupling is not needed. If the first stage loses its low root its capacity tends to one, and a comparison argument carries the second across in finite time whenever κ₂ exceeds a(Q)²/4, which reproduces the hypothesis for the third: recovery passes every stage whose coupling exceeds a(Q)²/4 and stops at the first that does not. Bisecting each coupling with the others held strong returned 0.0900000 at every stage, and the arrest depth was the first subcritical stage in every case (Fig. 3a). The sharp quantity is minₖ κₖ. Existence of the branch is different: for k ≥ 2 a low root exists whatever κₖ is, so only κ₁ can annihilate it, and κ_{c} is unchanged by the downstream couplings and by chain length (Supplementary Note 4).'),
 ('h2', 'Three coupling regimes, one of them metastable'),
 ('p', 'Because κ_{c} moves with a(Q), the coupling axis has three regimes rather than two (Fig. 2b). For κ > a₀²/4 a crossing propagates at once. For κ < a₁²/4 the branch survives at every attainable evidence level; only raising κ removes it. Between the two the system stalls after the crossing and then completes spontaneously: residual joint availability is small but not zero, so Q creeps upward, a(Q) falls, and the branch is annihilated when a(Q) drops below 2√κ. Inverting the threshold gives the evidence level at which that happens, the annihilation locus'),
 ('eq', '*Q∗(κ) = [ (a₀ − 2√κ) / (a₀ − a₁) ]*^{1/p}*,*'),
 ('p', 'which lies in (0,1) exactly on the intermediate band, tending to 1 as κ approaches a₁²/4 from above and to zero at a₀²/4. Locating that level in a full-system integration is not an independent test, the event function and the formula being the same discriminant. What the trajectory does next is: escape, detected on a different function, followed the predicted fold by 0.003% of the annihilation time at κ = 0.05 and 0.11% at κ = 0.08, and by 0.002% with the wide selector. Below the band nothing moved within 5 × 10^{10} time units.'),
 ('p', 'This changes the status of the partial state on the intermediate band and not only its fate. While the root is recovered the evidence coordinate increases monotonically and the residual availability is bounded below, so the locus is reached in finite time for every κ above a₁²/4, where the partial configuration is an equilibrium of the capacity subsystem alone and metastable in the full system. For κ < a₁²/4 the branch is never removed and the trajectory follows it as Q rises, ending in a partial attractor at Q = 1, the first stage creeping from 0.0050 to 0.0238 at κ = 0.003. Nor is that profile firmly held. The barrier the first stage must clear is 1.7 × 10^{−4}, and additive noise of standard deviation 0.008, under one per cent of the capacity range, took about 470 of 500 paths off the branch within 2 × 10^{4} time units, median of order 4 × 10^{3}; at 0.002 none left (Fig. 3b). Confinement is a statement about the deterministic field, and the regimes are better read as an ordering of timescales than as three fates.'),
 ('p', 'The band is stage-wise as well. Each stage carries its own locus Q∗(κₖ), but none can cross before its predecessor, so stage k completes when the evidence level first exceeds the running maximum of Q∗ up to k. Over five coupling vectors the fifteen completions all sat above that maximum, by 0 to 0.041, in chain order. The intervals between them are compressed, 0.004% to 9% of the wait preceding them, because a stage coming online raises C(r) and accelerates Q.'),
 ('p', 'The two statements about joint availability on which this rests refer to different timescales and are reconciled in Supplementary Note 3. The absolute waiting time inherits the evidence rate and the regularisation of ref. 4, differing by 380 between the selector scales, and is not a prediction here, whereas the band, its boundaries and the locus are.'),
 ('h2', 'Intermittent coupling acts as a duty-weighted average'),
 ('p', 'Any repeated treatment raises a parameter for a fraction of the time rather than holding it. The field is linear in κ, so the averaged field is exactly the field at the duty-weighted mean and the criterion is expected rather than discovered^{21}; the trajectory error is of the order of the modulation period and not zero. Modulating κ as a rectangular wave with duty fraction f from the partial branch, escape occurred when the duty-weighted mean exceeded κ_{c} and not otherwise, in 40 of 42 cells (Fig. 4a); the two exceptions had f = 0.05 and an elevated coupling at least fourteen times κ_{c}, where the low phase is long enough to relax back. Sweeping the period at a fixed mean of 1.2 κ_{c}, the modulated escape time tracked the average to within 1% below a hundredth of the static escape time, then fell to a sixth of it once the period reached that time (Fig. 4b). Averaging fails in both directions: the criterion holds only where deliveries are frequent compared with the escape they produce.'),
 ('h2', 'Escape from the partial state is slow and stepwise'),
 ('p', 'Why that exponent is one half can be read off the branch. On a forward chain the frozen capacity Jacobian is lower triangular, so its spectrum is its diagonal, and at the first stage, with κ = a r₁ − r₁², it collapses exactly to'),
 ('eq', '*λ₁ = − √(a² − 4κ) (1 − r₁),*'),
 ('p', 'confirmed numerically to 10^{−16}. Every eigenvalue is negative wherever the branch exists, so the partial configuration is stable at every point of it and the branch is normally hyperbolic away from the fold, where the trajectory tracks it to first order in ε^{22}; the fold itself is not covered. The rate that vanishes at the fold is the square root of the same discriminant that fixes κ_{c}, and vanishes nowhere else (Fig. 3c; Supplementary Note 5). Criterion and exponent are two readings of one expression, one asking when the discriminant vanishes and the other how fast.'),
 ('p', 'Above κ_{c} the escape time diverges as a power of the excess, and the exponent is one half. Extending the excess down to 10^{−5}, the local slope of log escape time against log excess falls monotonically from about 0.65 at an excess of 0.8 to 0.501 at the smallest examined (Fig. 4c); a single regression returns 0.529 over that range and 0.509 below 5% excess, both reporting curvature rather than the asymptote. In the full system, started from the exact branch at half the critical coupling, the same fits give 0.48 to 0.57 over the full range and 0.39 to 0.53 below 5%.'),
 ('p', 'Two effects are superimposed in that spread, and one control separates them. With the evidence rate set to zero and everything else unchanged, the six threshold levels collapse onto one another, the fit below 5% excess lying between 0.527 and 0.532 (Supplementary Note 2), so curvature is intrinsic to the fold. The spread is not: at ε = 0.02 the same fit falls to 0.386 at Q_{0} = 0.5, because the threshold moves while the system waits. The lowered exponent is evidence feedback, not asymptotic convergence.'),
 ('p', 'The shape of the trajectory matters more than the exponent. At an excess of 5% the first two downstream capacities crossed their own unstable roots at 0.31 and 0.65 of the escape interval and reached 0.9 within 3.3% of it, so each spends most of the interval below its own threshold and crosses quickly (Fig. 4d). The result is a staircase. We quote no single fraction for the quiet interval, since it moves from 0.59 to 0.96 as the rate threshold moves from 0.5% to 20% of the maximum. This is the sharpest observable difference between the axes: a crossing driven by the root excursion is fast, sharp and complete, an escape driven by coupling slow and stepwise.'),
 ('h2', 'Neither axis requires maintenance after the crossing'),
 ('p', 'At the recovered configuration all capacities are maximal, adjacent differences vanish and the coupling term contributes nothing, so it is asymptotically stable for every non-negative coupling including zero. A system carried past the last point of no return by either axis stays recovered when both return to their previous values, so durability after a course is not evidence of lasting change. What does change with sustained occupancy is Q, lowering the threshold from 0.60 towards 0.15 and widening the tolerated excursion Δ_{c}(Q) from 0.40 to 0.85. Relapse is a re-crossing of the same surface at low consolidated evidence, not decay of a treatment effect, a reading in tension with the reported value of maintenance treatment that the seventh prediction separates.'),
 ('h2', 'Identifying coupling and threshold from the profile'),
 ('p', 'The obstacle to testing this is that coupling has never been measured, and inferring it from failure to respond is circular. The first stage breaks the circle: its equilibrium condition factorises exactly, so the coupling follows from the first-stage ratio and the threshold,'),
 ('eq', '*κ = a r₁ − r₁²,    with  a = 1 − Δ_{c},*'),
 ('p', 'with Δ_{c}(Q) the largest downward excursion from full recovery absorbed at that evidence level. On the exact equilibrium this returns the coupling to machine precision, with error first order in the error of r₁. Eliminating κ between the first two stages instead gives a = (r₁³ − r₂²)/(r₁² − r₂), exact only on the stage-wise closed form from which it is derived; on the exact equilibrium it is biased before any noise is added, by −9.4% in the threshold and −11.2% in the coupling at κ = 0.003. It is also ill-conditioned, r₂ being smaller than r₁ by a factor of fifty: at 1% lognormal noise the recovered coupling has an interquartile range of ±10%, and at 10% it is uninformative (Fig. 2c). The first-stage estimator is therefore primary.'),
 ('p', 'This is what makes the two axes distinguishable in practice. An intervention acting only on the capacity state leaves the resting profile of a partial branch unchanged; one acting on coupling moves it, by a calculable amount, without the patient recovering. Two inputs are needed and neither is available. The estimator uses r₁ itself and not a monotone transform, so it needs an index of the first downstream capacity on its own [0,1] scale, anchored at zero for complete failure and at one for the fully recovered level in the same patient. It also needs the threshold prevailing on the branch, a(Q), which is not what an acute challenge reads: a graded challenge probes the collapsed configuration and estimates the depleted-evidence threshold a₀, whereas the branch sits at consolidated evidence, and the two differ fourfold here. The substitution is not a small bias. At κ = 0.003 the estimator returns 2.19 times the true coupling if a = 0.30 is used and 4.56 times it at a₀ = 0.60, against 12% for a 10% error in the correct threshold. The prevailing threshold must be obtained at the evidence level of the branch, jointly with Q, and we specify no validated procedure. The coupling is structurally identifiable and not yet operationally measurable.'),
 ('h2', 'What the geometry does and does not say about ECT'),
 ('p', 'The results above are statements about the model, and their clinical use needs a representation, stated here as a hypothesis rather than derived. We take ECT to act as an excursion on the root and, over a course, as an increase of the effective coupling. The halves are not equally secure. A single treatment is a large acute exogenous perturbation, so its representation by u is close to direct. The coupling half is a coarse-grained hypothesis about propagation from one capacity to the next, not an identification with synaptic strength or with connectivity between regions.'),
 ('p', 'Under that representation three clinical regularities follow from the geometry rather than requiring separate accommodations: the response to a graded stimulus is all-or-none, the shape of the lorazepam challenge^{8,9}; response after benzodiazepine failure needs only a larger excursion; and conversion can be fast, because beyond the surface the chain converges on the ordinary capacity timescale^{9,10}.'),
 ('p', 'The two axes are not competing accounts of one treatment but two states in which one treatment cannot be doing the same thing, and the geometry turns that into a conditional. If the coarse-grained coupling is diffusive, and if the threshold is not itself altered, then no coupling increase converts a wholly sub-threshold configuration, so an acute conversion requires an excursion-like component. On the modelled partial branch a partial converse holds: the root is already maximal, the excursion axis saturated and the evidence coordinate closed, so a durable change of the resting profile there is a change of coupling. Both statements are conditional on the coupling form, on the architecture, and on identifying the acute and chronic presentations with the collapsed configuration and the branch. None of these is shown here, and the alternatives to a coupling change are set out next.'),
 ('p', 'Within the architecture, three things other than a change of coupling could still take a patient off the branch: an input driving a downstream capacity above its own unstable root, noise, which the shallow barrier admits, and spontaneous completion on the band at a time the treatment did not set. The resting profile separates all three, since only a change in κ moves the resting first stage and keeps it moved.'),
 ('blank', ''),
 ('h1', 'Discussion'),
 ('p', 'Of the three quantities we allow an intervention to move, one has no external port and the other two are not alternatives, each powerless exactly where the other is decisive. The criterion κ_{c} = a²/4 carries the rest, exact because the first stage factorises, applied stage by stage, and, read as the discriminant whose square root is the rate vanishing at the fold, it fixes the escape exponent too. What is exact and what is not should be kept apart: the first stage, the criterion, the locus, the spectrum and the estimator are exact; the stage-wise closed form downstream is an approximation with residual κ rₖ(rₖ₋₁ − 1), inherited as a bias by the two-ratio inversion.'),
 ('p', 'We are equally explicit about the clinical bridge. Representing electroconvulsive therapy as an excursion with a coupling component is a hypothesis, asymmetric in its security, and nothing here identifies which physiology realises either axis. The geometry supplies a conditional necessary condition on each side; the resting profile would turn it into a measurement, once both of its inputs can be read. One weakness is better named than left implicit. The whole division rests on the coarse-grained coupling being diffusive, and the one biological observation we have appealed to, an increase in co-firing within cell assemblies^{15}, points instead to the additive form, which the theorem does not cover.'),
 ('p', 'The account is falsifiable in seven places. First, acute conversion should be all-or-none in the excursion coordinate, with no group whose recovery is partial in the amplitude delivered; this is testable now, from dose-response data in existing lorazepam-challenge cohorts, and it separates the coupling forms as well as the axes, since a diffusive coupling makes the excursion necessary and an additive drive does not. Second, non-response to a smaller excursion and response to a larger should need no patient property beyond amplitude. Third, response kinetics should fall into three classes, fast complete conversion, a stalled state that completes late unaided, and one that does not, with membership predicted by the first-stage ratio through a₁²/4 and a₀²/4. Fourth, within the slow class, frequency and per-session amplitude should be interchangeable between those boundaries. Fifth, on the branch the resting first-stage ratio should be unchanged by an intervention acting on the capacity state and changed by one acting on coupling. Sixth, arrest depth should identify the weak stage, and late completions should arrive in bursts rather than evenly. Seventh, relapse risk should track occupancy of the recovered configuration rather than the number of treatments.'),
 ('p', 'Several limitations qualify these results. The confinement theorem holds for arbitrary graphs and unequal thresholds but only for diffusive coupling, and everything after the crossing assumes a forward chain of four stages, a length not varied for the staircase, the exponent or the modulation. Which quantities are externally accessible is an assumption. The identification needs both an anchored index of the first downstream capacity and the threshold prevailing on the branch, and no published measure supplies either; the two state identifications are assumed; the dependency order rests on a single unblinded series, and the proportional recovery rule in stroke warns that apparent ordering can arise from ceiling effects and mathematical coupling^{23}. The structural statements hold for any parameters; the boundaries 0.60, 0.090 and 0.0056 inherit the illustrative set of ref. 4. Model time is not clinical time, and the waiting time moves by 380 with the regularisation, so timing statements should be read as orderings. Whether the fold is passed with the delay of a dynamic saddle node is left open.'),
 ('blank', ''),
 ('h1', 'Conclusion'),
 ('p', 'Within the intervention architecture set out here, three quantities can be moved, and this work fixes what each can do. Evidence has no external port. Non-negative diffusive coupling cannot produce a crossing out of a configuration in which nothing is above threshold, on any graph, with unequal weights and thresholds and in either direction, so under that coupling form an acute conversion requires an excursion; an additive excitatory drive is a different model and converts on its own. After the crossing one exact criterion, a(Q)²/4, applied stage by stage, sets how far recovery reaches and when each stage completes, and the same discriminant gives the spectrum on the branch and the square-root escape law. The first stage inverts exactly, which is what would make the division of labour testable. For a treatment used in both states the geometry predicts an excursion component acutely and a coupling component on the branch, conditional on the coupling form and on the two state identifications.'),
 ('blank', ''),
 ('h1', 'Methods'),
 ('h2', 'Model and reduction'),
 ('p', 'The system is that of ref. 4, with equal coupling across stages except where stated. The root obeys a bistable law plus an external excursion and each downstream capacity obeys the same law plus a diffusive term proportional to the difference from its immediate predecessor. The bistable law is a cubic with stable states at zero and one and an unstable threshold decreasing convexly with the evidence coordinate from a₀ at no evidence to a₁ at full evidence. Except where stated, analyses were run on the capacity subsystem with the threshold held at a fixed value: this is exact on the partial branch, where the joint availability that drives the evidence coordinate is of order 10^{−10}, and conservative from a collapsed state, where the root lies below threshold and the evidence channel only attenuates. The full five-dimensional system was used for the intermediate-regime integration and for the confirmatory exponent measurement, with the illustrative parameters of ref. 4 throughout: a₀ = 0.60, a₁ = 0.15, p = 2, ε = 0.02, ρ = 1 and the one-sided regularisation χ_{±}(z) = exp(−s/z²) on its own half-line, run at both scales used there, s = 0.01 and s = 1. Integration used an adaptive stiff solver with relative tolerance 10^{−10} and absolute tolerance 10^{−13}, with event detection for crossings.'),
 ('h2', 'Confinement below threshold'),
 ('p', 'One hundred systems were drawn from each of four graph families, feed-forward chains, symmetric graphs, purely reverse chains and general directed graphs, on three to six nodes, with weights drawn from a mixture placing mass at zero, 0.5, 3 and 30 times a uniform draw on (0,1]. The same mixture was used for every family, restricted to the sub-diagonal for feed-forward chains, symmetrised for symmetric graphs and transposed into the strict upper triangle for purely reverse chains, so no family was given a narrower weight distribution than another. Every capacity was initialised below a₀ and the system integrated with no external input; recorded quantities were whether any capacity reached the threshold and the largest increase of maxₖ rₖ over its initial value. Separately, in the forward chain, root trajectories from the same sub-threshold initial condition were computed at κ = 0.6, 3.0 and 30.0 on a common dense grid and compared pointwise rather than at the endpoint. The reverse-coupling counterexample used a bidirectional chain from r = (0.30, 0.90, 0.90, 0.90), at forward weight 0.01 with reverse weight 0.10 and with reverse coupling alone. A further 200 general directed graphs were drawn with a threshold per node from the uniform distribution on [a₁,a₀] and every capacity initialised below the smallest of them, which is the hypothesis the heterogeneous form of the principle needs. The scope experiments of Supplementary Note 6 used the same four-capacity system started at 0.30 everywhere with no external input: an additive form drₖ/dt = rₖ(1 − rₖ)(rₖ − a) + w(1 − rₖ)Σ_{j≠k} S(rⱼ) on the complete graph with S(r) = r and with S sigmoidal of midpoint 0.30 and width 0.05, the critical weight bracketed by bisection on the final maximum; the same start under diffusive coupling at weights 0.1, 1, 10 and 100; the same start under the reference coupling at thresholds a = 0.60, 0.40, 0.30 and 0.25; and a root placed at 0.61 with the rest at zero, integrated to 3 × 10⁴ at couplings from 0.6 to 0.01.'),
 ('h2', 'Root pulse and excursion sweep'),
 ('p', 'Pulses were applied as u(t) = A on [0,τ] and zero afterwards, from the collapsed chain at Q = 0, at durations τ = 0.25, 0.5, 1, 2 and 5. For each duration the critical amplitude was bracketed by bisection, and outcomes recorded at 0.8, 0.99, 1.01 and 1.2 times it together with the root value reached at the end of the pulse. A separate sweep set the root directly to a value v with no input, over a grid spanning a₀ and refined near it; v = a₀ lies on the invariant surface and was excluded from the grid rather than integrated, since the outcome there is neither corner. The transition was located by bracketed root-finding at three coupling values.'),
 ('h2', 'Partial branch, critical coupling, identification'),
 ('p', 'The first stage was taken from the factorised quadratic, which is exact. Each downstream stage was obtained as the low root of its cubic, bracketed on (0,a) because the cubic is positive at 0 and negative at a whenever 0 < rₖ₋₁ < a, and solved to machine precision; the residual of the vector field at that profile was below 10^{−18}, and a perturbation of 10^{−4} in every downstream capacity returned to it. The stage-wise closed form was retained only to quantify its departure, reported as a relative error against the exact profile and against the analytic residual κ rₖ(rₖ₋₁ − 1). The propagation boundary was located by bracketed root-finding on the coupling at which a supracritical excursion reaches the distal capacity. In the full system the fold was located as a separate non-terminal event at a(Q)² − 4κ = 0; because that event function is the discriminant from which the locus is derived, the agreement it reports is a check on the solver rather than a test of the formula, and the independent comparison is with the escape event, detected on r₃ − a(Q) = 0. Identification was evaluated on the exact equilibrium, for the first-stage estimator and for the two-ratio inversion, without noise and then under multiplicative lognormal noise of coefficient of variation 1%, 5% and 10%, two thousand draws per level, reporting medians and interquartile ranges.'),
 ('h2', 'Unequal coupling, spectrum and noise'),
 ('p', 'For unequal coupling each stage was given its own κₖ, and the boundary of a single stage was bracketed by bisection on that stage with the others held at 1.0, from the same supracritical excursion used for the equal-coupling boundary. The constants of the comparison lemma of Supplementary Note 4 were evaluated on a grid of 20001 points on [0,0.9] and the bound on the crossing time checked against the integrated comparison equation at four couplings above a₀²/4. Arrest depth was read from seven coupling vectors as the number of downstream capacities above 0.5 at the end of the integration, and compared with the index of the first stage whose coupling does not exceed a(Q)²/4. Existence of the branch was tested at κ₁ = 0.005625 and 0.00563 with downstream couplings spanning 10^{−4} to 30, and the critical coupling was located by bisection to 10^{−14} at chain lengths from one stage to twelve. The spectrum was taken as the diagonal of the capacity Jacobian, which is lower triangular on a forward chain, evaluated on the exact branch at three thresholds and ten couplings per threshold and compared with the closed form. For the noise runs the capacity block was integrated by Euler-Maruyama with additive Wiener increments of standard deviation σ on each capacity, step 0.05, the root held at 1, 500 paths per σ and a horizon of 2 × 10^{4}; escape was the first stage crossing its own unstable root, and the barrier was computed from the potential of the first-stage field.'),
 ('h2', 'Intermittent modulation and period sensitivity'),
 ('p', 'The coupling was modulated as a rectangular wave between a base value of 0.003 and an elevated value, with duty fraction f, starting from the exact partial branch. Escape was scored as the most distal capacity crossing the unstable root of its own bistable law. For the criterion grid the period was one fiftieth of the static escape time at a duty-weighted mean of 1.2 κ_{c}. For the period sweep the duty fraction and elevated value were fixed so that the duty-weighted mean equalled 1.2 κ_{c}, and the period was varied from 10^{−3} to 5 times that static escape time; runs in which the escape completed inside the first elevated phase were flagged, since the ratio saturates there and successive points are not independent.'),
 ('h2', 'Escape-time scaling'),
 ('p', 'Escape times were recorded at excesses from 10^{−5} to 0.8 above the critical coupling. The local slope was computed between consecutive excesses, and regressions were also taken over the full range and restricted to excesses below 5%. Reported escape times are for the most distal capacity crossing the unstable root of its own law, which is the definition of the escape time and not the completion of recovery. The measurement was repeated in the full system at Q = 0, 0.5, 0.7, 0.9, 0.99 and 1, in each case starting from the exact partial branch at half the critical coupling for the prevailing threshold, and then repeated again with the evidence rate set to zero and every other setting, including the starting point and the excess grid, left unchanged; that control isolates the contribution of evidence feedback, and it was extended to an excess of 2 × 10^{−4} at Q = 0 and Q = 0.5 to locate the asymptote. The fraction of the escape interval showing no appreciable change was computed at rate thresholds from 0.5% to 20% of the maximum rate and is reported as a range rather than a single value.'),
 ('h2', 'Ethics and reporting'),
 ('p', 'This is a theoretical and numerical study. No human participants, animals, tissue or clinical records were involved and no ethics oversight was required. Clinical quantities cited in the text are taken from the published literature and are used to state what a model must reproduce, not as data analysed here.'),
 ('blank', ''),
 ('h1', 'Supplementary information'),
 ('p', 'Supplementary Note 1 restates, with proofs, the four properties of the underlying geometry used here: asymptotic stability of the recovered configuration for every non-negative coupling, asymptotic stability of the failed configuration for positive attenuation, exact invariance of the switching surface S = {r₀ = a(Q)} with forward invariance of z > 0 and z < 0, and the sufficient bound K* above which every trajectory off S reaches a corner. It is provided so that the present results can be checked without recourse to ref. 4, and it also records the analytic residual of the stage-wise closed form. Supplementary Note 2 reports the zero-evidence-rate control for the escape exponent, Supplementary Note 3 reconciles the two statements about joint availability on the partial branch, and Supplementary Notes 4 and 5 prove the two facts used for unequal coupling and the spectrum on the branch. Supplementary Note 6 gives the three experiments that delimit the confinement statement: additive excitatory coupling, lowering of the threshold, and propagation below a₀²/4.'),
 ('h1', 'Data availability'),
 ('p', 'No patient data were used in this study. All quantities reported here are generated by the deposited code from the parameter set specified in Methods, and no other data underlie the findings. Source data for Figs. 1 to 4 are provided with the paper as a single workbook with one sheet per panel, written by the same script that draws the figures.'),
 ('h1', 'Code availability'),
 ('p', 'Code reproducing every reported value and figure is provided as a single self-contained script that also re-checks each reported number against the stored results. It covers the confinement sweeps over the four graph families and over heterogeneous thresholds, the reverse-coupling counterexample, the rectangular-pulse and excursion grids, the exact partial branch and its identification, the unequal-coupling and chain-length sweeps, the spectrum on the branch, the intermittent-modulation grid and period sweep, the escape-time scaling with its zero-evidence-rate control, and the noise runs. Running it end to end writes the results file, the four figures and the source-data workbook. The script also carries the scope experiments of Supplementary Note 6, the compliance audit of the manuscript, and a self-check that every value printed in the text agrees with the stored results at the precision printed. It is available at https://github.com/entrance4-png/catatonia-two-axes and archived at Zenodo under https://doi.org/10.5281/zenodo.21880298, which resolves to the latest version, alongside the code accompanying the model on which this work builds^{4}.'),
 ('blank', ''),
 ('h1', 'References'),
 ('p', '1. Saito, H. *et al.* Recovery from catatonia follows a hierarchical precision order. Preprint at medRxiv https://doi.org/10.1101/[medRxiv DOI to be inserted at submission] (2026).'),
 ('p', '2. Saito, H. The Saito Loop: a precision-hierarchical account of catatonia. Preprint at https://osf.io/preprints/psyarxiv/de6bu_v1 (2026).'),
 ('p', '3. Saito, H. Recovery order in adaptive systems is set by dependency structure. Preprint at https://osf.io/preprints/psyarxiv/8ae7w_v1 (2026).'),
 ('p', '4. Saito, H. Evidence-coupled bistability converts dependency structure into a recovery geometry. Preprint at Research Square https://doi.org/10.21203/rs.3.rs-10620384/v1 (2026).'),
 ('p', '5. Northoff, G. What catatonia can tell us about top-down modulation: a neuropsychiatric hypothesis. *Behav. Brain Sci.* **25**, 555–577 (2002).'),
 ('p', '6. Hirjak, D. *et al.* Catatonia. *Nat. Rev. Dis. Primers* **10**, 49 (2024).'),
 ('p', '7. Rogers, J. P. *et al.* Evidence-based consensus guidelines for the management of catatonia: recommendations from the British Association for Psychopharmacology. *J. Psychopharmacol.* **37**, 327–369 (2023).'),
 ('p', '8. Bush, G., Fink, M., Petrides, G., Dowling, F. & Francis, A. Catatonia. II. Treatment with lorazepam and electroconvulsive therapy. *Acta Psychiatr. Scand.* **93**, 137–143 (1996).'),
 ('p', '9. Sienaert, P., Dhossche, D. M., Vancampfort, D., De Hert, M. & Gazdag, G. A clinical review of the treatment of catatonia. *Front. Psychiatry* **5**, 181 (2014).'),
 ('p', '10. Lloyd, J. R., Silverman, E. R., Kugler, J. L. & Cooper, J. J. Electroconvulsive therapy for patients with catatonia: current perspectives. *Neuropsychiatr. Dis. Treat.* **16**, 2191–2208 (2020).'),
 ('p', '11. Pelzer, A. C. M., van der Heijden, F. M. M. A. & den Boer, E. Systematic review of catatonia treatment. *Neuropsychiatr. Dis. Treat.* **14**, 317–326 (2018).'),
 ('p', '12. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P. & Pezzulo, G. Active inference: a process theory. *Neural Comput.* **29**, 1–49 (2017).'),
 ('p', '13. Chueh, K. N., Conley, C. C. & Smoller, J. A. Positively invariant regions for systems of nonlinear diffusion equations. *Indiana Univ. Math. J.* **26**, 373–392 (1977).'),
 ('p', '14. Smith, H. L. *Monotone Dynamical Systems: An Introduction to the Theory of Competitive and Cooperative Systems* (American Mathematical Society, 1995).'),
 ('p', '15. Moore, H. *et al.* Stimulation modulates gene-linked cell assemblies in the human brain. *Nature* https://doi.org/10.1038/s41586-026-10879-9 (2026).'),
 ('p', '16. Kouvaris, N. E., Kori, H. & Mikhailov, A. S. Traveling and pinned fronts in bistable reaction-diffusion systems on networks. *PLoS ONE* **7**, e45029 (2012).'),
 ('p', '17. Keener, J. P. Propagation and its failure in coupled systems of discrete excitable cells. *SIAM J. Appl. Math.* **47**, 556–572 (1987).'),
 ('p', '18. Mallet-Paret, J. The global structure of traveling waves in spatially discrete dynamical systems. *J. Dyn. Differ. Equ.* **11**, 49–127 (1999).'),
 ('p', '19. Elmer, C. E. & Van Vleck, E. S. Spatially discrete FitzHugh-Nagumo equations. *SIAM J. Appl. Math.* **65**, 1153–1174 (2005).'),
 ('p', '20. Strogatz, S. H. *Nonlinear Dynamics and Chaos* 2nd edn (Westview, 2015).'),
 ('p', '21. Sanders, J. A., Verhulst, F. & Murdock, J. *Averaging Methods in Nonlinear Dynamical Systems* 2nd edn (Springer, 2007).'),
 ('p', '22. Fenichel, N. Geometric singular perturbation theory for ordinary differential equations. *J. Differ. Equ.* **31**, 53–98 (1979).'),
 ('p', '23. Hope, T. M. H. *et al.* Recovery after stroke: not so proportional after all? *Brain* **142**, 15–22 (2019).'),
 ('blank', ''),
 ('h1', 'Acknowledgements'),
 ('p', 'The author thanks colleagues at Onda-daini Hospital, Nippon Medical School and Karolinska Institutet for discussion. No funding was received for this work.'),
 ('h1', 'Author contributions'),
 ('p', 'H.S. conceived the study, derived and implemented the model, performed the numerical analysis, and wrote the manuscript.'),
 ('h1', 'Competing interests'),
 ('p', 'The author declares no competing interests.'),
 ('blank', ''),
 ('pagebreak', ''),
 ('h1', 'Figures'),
 ('blank', ''),
 ('fig', 'Fig1.png'),
 ('leg', '**Fig. 1 | The crossing is set by the root excursion and is independent of coupling.** **a**, The chain. In catatonia r₀ to r₃ are sensory, policy, motivational and fast-volatility precision, and Q is the identifiability of slow-volatility precision. External input enters the root alone; coupling acts on differences between adjacent capacities and is absent from the root equation. **b**, The running maximum maxₖ rₖ(t) from a sub-threshold start, one example per graph family. Across 400 systems drawn from the four families no trajectory reached the threshold and the maximum never rose. **c**, Rectangular pulses u(t) = A on [0,τ] from the collapsed chain, at five durations and four amplitudes each. Outcome is fixed by the root value at the end of the pulse alone: conversion occurs exactly when r₀(τ) > a₀. **d**, With the same supracritical excursion in every run, the outcome after the crossing is set by coupling, with a sharp boundary at a₀²/4 = 0.090 located at 0.0900005.'),
 ('pagebreak', ''),
 ('fig', 'Fig2.png'),
 ('leg', '**Fig. 2 | The partial branch, its removal, and identification from the resting profile.** **a**, Partial branch against coupling at consolidated evidence: exact equilibrium in colour, stage-wise closed form in grey. The two coincide at the first stage and diverge downstream. Inset, relative departure of the closed form from the exact profile: identically zero at r₁, rising to 2.0% at r₂ and 4.0% at r₃ at κ = 0.003 and growing as the coupling falls. The first-stage roots collide at κ_{c} = a₁²/4 = 0.0056. **b**, Because κ_{c} = a(Q)²/4 falls as evidence consolidates, the coupling axis has three regimes: immediate propagation above a₀²/4, deterministic confinement to the branch below a₁²/4, and an intermediate band in which the partial state is metastable, stalling and then completing without further intervention. Read against the upper axis the same curve is the annihilation locus Q∗(κ); markers show the three couplings integrated in the full system. **c**, Recovered coupling on the exact equilibrium. The first-stage estimator is unbiased at zero noise and degrades gracefully; the two-ratio inversion is biased before any noise is added and is ill-conditioned. Points are medians and bars interquartile ranges over two thousand draws per level.'),
 ('pagebreak', ''),
 ('fig', 'Fig3.png'),
 ('leg', '**Fig. 3 | The weakest stage sets where recovery stops, and the branch is shallowly held but hyperbolically stable.** **a**, One stage swept while its neighbours stay strong. Recovery stops at the swept stage and nowhere else, and it stops exactly at a₀²/4, so the criterion applies stage by stage and the sharp quantity is the weakest coupling. **b**, Fraction of 500 noisy paths still on the confined branch at κ = 0.003, where the barrier is 1.7 × 10^{−4}. Additive noise of standard deviation 0.008 empties the well within the horizon; 0.002 does not. **c**, Decay rates on the branch at consolidated evidence against the distance from the fold. The rate that vanishes at the fold follows the closed form √(a² − 4κ)(1 − r₁) exactly and does so as a square root, a straight line of slope 1/2 here, while the second stage stays fast.'),
 ('pagebreak', ''),
 ('fig', 'Fig4.png'),
 ('leg', '**Fig. 4 | Intermittent delivery, the escape exponent, and the stepwise escape.** **a**, Escape from the partial branch under rectangular modulation follows the duty-weighted criterion in 40 of 42 combinations; the exceptions are large amplitude at low duty, where the fast-modulation assumption fails. **b**, Holding the duty-weighted mean at 1.2 κ_{c}, the modulated escape time tracks the averaged prediction for periods below a few per cent of the escape time and then falls below it, so averaging understates efficacy there, the reverse of a. Diamonds mark points at which the escape completes inside the first elevated phase, where the ratio saturates. **c**, Local slope of log escape time against log excess in the capacity subsystem, converging to 0.501 at the smallest excess examined; the dashed line is 1/2. Inset, exponent fitted below 5% excess in the full system against the prevailing threshold, with the evidence channel on and off. With the channel off the six levels collapse; the depression at ε = 0.02 is therefore evidence feedback and not the approach to the asymptote. **d**, At an excess of 5% the first two downstream capacities cross at 0.31 and 0.65 of the escape interval and reach 0.9 within 3.3% of it in both cases. The third crosses at 1.00 by construction, since the escape time is defined as that crossing.'),
]


SUPPLEMENT_TITLE = "Supplementary Information"

SUPPLEMENT_PARAS = [
 ('title', SUPPLEMENT_TITLE),
 ('blank', ''),
 ('p', '**Catatonia recovery separates two intervention axes that constrain electroconvulsive therapy**'),
 ('p', 'Hiroki Saito'),
 ('blank', ''),
 ('h1', 'Supplementary Note 1. Properties of the underlying geometry'),
 ('p', 'The main text uses four properties of the evidence-coupled bistable cascade. They are established in ref. 4 of the main text; they are restated and proved here, in the notation of the main text, so that the results of this paper can be checked without recourse to that work. Throughout, g(r,Q) = r(1 − r)(r − a(Q)), the coupling is non-negative, the root carries no incoming coupling, and the selectors χ₊ and χ₋ are non-negative, locally Lipschitz, and vanish on z ≤ 0 and z ≥ 0 respectively, so in particular χ₊(0) = χ₋(0) = 0.'),
 ('h2', 'S1. The recovered configuration is stable for every coupling'),
 ('p', 'At E_{rec} = (1,1,1,1) with any Q, g(1,Q) = 0 and every difference rₖ₋₁ − rₖ vanishes, so E_{rec} is an equilibrium of the capacity block for every κ ≥ 0. Differentiating g gives ∂g/∂r = (1 − 2r)(r − a) + r(1 − r), which at r = 1 equals −(1 − a) < 0 since a < 1. The capacity Jacobian is lower triangular, because stage k receives only from stage k − 1, with diagonal entries −(1 − a) − κₖ, all strictly negative. In the evidence direction the factor 1 − Q multiplies the only positive term, so the linearisation there is −ε χ₊(z) C(r) ≤ 0, strictly negative while z > 0. The configuration is therefore asymptotically stable, and the conclusion does not depend on the value of the coupling, including κ = 0.'),
 ('h2', 'S2. The failed configuration is stable when evidence attenuates'),
 ('p', 'At E_{path} = (0,0,0,0) the same computation gives ∂g/∂r = −a at r = 0, and the capacity Jacobian is again lower triangular with diagonal −a − κₖ < 0. There z = −a(Q) < 0, so χ₋ > 0 and the evidence equation reduces to dQ/dt = −ε ρ χ₋(z) Q, whose linearisation at Q = 0 is −ε ρ χ₋(−a) < 0 for ρ > 0. The failed configuration is then hyperbolically asymptotically stable. At ρ = 0 that eigenvalue vanishes and the set {(0,0,0,0,Q)} is a continuum of equilibria, Lyapunov stable in the capacities but not asymptotically stable in the full system.'),
 ('h2', 'S3. The switching surface is exactly invariant'),
 ('p', 'Let S = {r₀ = a(Q)} and z = r₀ − a(Q). On S, and with no external input, g(r₀,Q) = r₀(1 − r₀)(r₀ − a(Q)) = 0, so dr₀/dt = 0. Also z = 0 gives χ₊(z) = χ₋(z) = 0, so dQ/dt = 0. Hence'),
 ('eq', '*dz/dt = dr₀/dt − a′(Q) dQ/dt = 0    on S,*'),
 ('p', 'for every ε and every ρ. The surface is therefore invariant, and because the field is locally Lipschitz, solutions are unique, so a trajectory in {z > 0} cannot reach {z < 0} without spending time on S, which is impossible. Both regions are forward invariant. The defining relation r₀ = a(Q) contains neither ε nor ρ, so the position of S is independent of the evidence timescale. Two remarks bear on the main text. First, the argument uses χ₊(0) = χ₋(0) = 0; if the selectors were replaced by plain averaging, χ_{±}(0) = 1/2, the surface would move with the evidence timescale. Second, u(t) enters dr₀/dt directly, so an external input is exactly the term that can carry a trajectory off S, which is the sense in which the crossing belongs to that axis alone.'),
 ('h2', 'S4. A sufficient coupling bound for whole-chain convergence'),
 ('p', 'Ref. 4 establishes that if κₖ exceeds K* = max over a in [a₁,a₀] of (1 − a + a²)/3 then every trajectory off S converges to a corner, E_{rec} when z(0) > 0 and E_{path} when z(0) < 0. At the illustrative parameters K* ≈ 0.291. We quote this result rather than reprove it, and we do not rely on it: the main text uses it only for comparison, since for the specific question of whether a crossing at the root propagates to the distal capacity the sharp boundary is available in closed form and equals a(Q)²/4, which at depleted evidence is 0.090, a factor of three below K*. K* is sufficient for a stronger statement, convergence from every initial condition, and is correspondingly conservative here.'),
 ('h2', 'S5. Residual of the stage-wise closed form'),
 ('p', 'The main text uses the exact first-stage root and solves the downstream stages numerically. For completeness, the residual left by the stage-wise closed form is analytic. Suppose rₖ satisfies rₖ² − a rₖ + κ rₖ₋₁ = 0, that is rₖ(rₖ − a) = −κ rₖ₋₁. Substituting into the true equilibrium condition gives'),
 ('eq', '*rₖ(1 − rₖ)(rₖ − a) + κ(rₖ₋₁ − rₖ) = (1 − rₖ)(−κ rₖ₋₁) + κ rₖ₋₁ − κ rₖ = κ rₖ(rₖ₋₁ − 1),*'),
 ('p', 'which vanishes only at rₖ₋₁ = 1. The first stage is therefore exact and every later stage is not. At the illustrative parameters and κ = 0.003 the closed form overstates r₂ by 2.0% and r₃ by 4.0%, and the residual it leaves, −1.396 × 10^{−6} at the second stage, is analytic rather than numerical. Because 0 < rₖ₋₁ < a, the exact cubic rₖ(1 − rₖ)(rₖ − a) + κ(rₖ₋₁ − rₖ) is positive at rₖ = 0 and negative at rₖ = a, so its low root is bracketed and can be obtained to machine precision, which is what the deposited code does.'),
 ('blank', ''),
 ('h1', 'Supplementary Note 2. Zero-evidence-rate control for the escape-time exponent'),
 ('p', 'The escape time above the critical coupling diverges as a power of the excess with exponent one half, and two separate effects push a fitted exponent away from that value. The first is pre-asymptotic curvature: at any finite excess the fit still contains the approach to the asymptote, so it exceeds one half over a wide range and falls towards one half as the excess is reduced. The second is feedback through the evidence coordinate: while the system waits on the branch, Q rises, a(Q) falls and the critical coupling moves, so the excess against which the fit is taken is not the excess the system experiences. The first effect is present in the capacity subsystem, which has no evidence channel; the second exists only at a positive evidence rate.'),
 ('p', 'The control separates them by a single change. The full five-dimensional sweep is repeated with ε set to zero and everything else, including the six evidence levels, the starting point at half the critical coupling for the prevailing threshold, the excess grid and the solver settings, held fixed. With the evidence channel off the six levels must give one exponent, because the capacity block then depends on the evidence level only through the constant a, and the excess is measured in units of the corresponding κ_{c}.'),
 ('p', '**Exponents of the full system with the evidence channel on and off.** Each entry is the negative slope of log escape time against log excess, fitted over the whole grid (0.005 to 0.8) and restricted to excesses at or below 0.05. Columns: Q₀, a(Q₀), then full range and small excess at ε = 0.02, then full range and small excess at ε = 0.'),
 ('p', 'Q₀ = 1.00;  a = 0.1500;  ε = 0.02: 0.570, 0.532;  ε = 0: 0.570, 0.532'),
 ('p', 'Q₀ = 0.99;  a = 0.1590;  ε = 0.02: 0.530, 0.456;  ε = 0: 0.570, 0.532'),
 ('p', 'Q₀ = 0.90;  a = 0.2355;  ε = 0.02: 0.501, 0.432;  ε = 0: 0.571, 0.532'),
 ('p', 'Q₀ = 0.70;  a = 0.3795;  ε = 0.02: 0.484, 0.401;  ε = 0: 0.572, 0.531'),
 ('p', 'Q₀ = 0.50;  a = 0.4875;  ε = 0.02: 0.478, 0.386;  ε = 0: 0.574, 0.529'),
 ('p', 'Q₀ = 0.00;  a = 0.6000;  ε = 0.02: 0.516, 0.430;  ε = 0: 0.577, 0.527'),
 ('p', 'The collapse is exact to the third decimal: with the channel off the full-range exponent lies between 0.570 and 0.577 and the small-excess exponent between 0.527 and 0.532, a spread of 0.005 across a threshold range of a factor of four, against a spread of 0.146 in the same column with the channel on. At Q₀ = 1 the two sweeps agree to every digit reported, because z > 0 on the branch makes the only inflow term proportional to 1 − Q, which vanishes there, so the evidence coordinate is stationary and ε cannot act.'),
 ('p', 'Continuing the control to smaller excess confirms that the residual curvature is the asymptote and nothing else. At Q₀ = 0.5 the local slope between consecutive excesses is 0.512 at 3.2 × 10^{−3}, 0.508 at 1.4 × 10^{−3}, 0.505 at 7.1 × 10^{−4} and 0.504 at 3.2 × 10^{−4}; at Q₀ = 0 the same sequence is 0.510, 0.507, 0.505 and 0.503. Both approach one half from above, at the same rate, at thresholds differing by a factor of 1.23.'),
 ('p', 'One alternative explanation is excluded arithmetically rather than numerically. Every sweep starts on the exact partial branch at κ = κ_{c}/2, and the first stage there is r₁ = a(1 − 1/√2)/2, while the branch remnant at κ_{c} is r₁ = a/2. The ratio is 1 − 1/√2 = 0.2929 at every threshold, since a cancels. The starting point is therefore the same fraction of the way to the fold at every evidence level and cannot produce a spread across levels.'),
 ('blank', ''),
 ('h1', 'Supplementary Note 3. The two statements about joint availability'),
 ('p', 'The main text uses the joint availability C(r) = ∏ⱼ rⱼ in two apparently opposite ways: it is treated as negligible when the capacity subsystem is analysed at a frozen threshold, and as the quantity that eventually destroys the partial branch. Both are correct, and they refer to different timescales.'),
 ('p', 'On the partial branch at the illustrative κ = 0.003 the exact profile is r = (1.000, 2.377 × 10^{−2}, 4.676 × 10^{−4}, 9.170 × 10^{−6}), so C(r) = 1.02 × 10^{−10}. The evidence equation gives |dQ/dt| ≤ ε C(r) = 2 × 10^{−12} while the root is recovered, so over an escape interval of a few hundred time units Q changes by less than 10^{−9} and a(Q) by less than 10^{−9}. Holding the threshold fixed over such an interval is therefore not an approximation at any digit the main text reports, which is what licenses the frozen reduction used for the branch, the critical coupling, the identification and the modulation results.'),
 ('p', 'The same residual availability is nevertheless bounded away from zero, and Q is monotone while z > 0, so the integral of the rate does not converge: given long enough, Q reaches any level below 1. The relevant level is the annihilation locus Q∗(κ), and the time to reach it is set by ε C(r) and by the selector scale. In the full-system integrations it ranges from 8.8 × 10^{5} time units at κ = 0.08 with the narrow selector to 3.4 × 10^{8} at the same coupling with the wide one, a factor of 380, and to 4.9 × 10^{7} at κ = 0.05 with the narrow selector. Below a₁²/4 the locus does not exist and no waiting time is defined.'),
 ('p', 'The consequence for the main text is a division between what is and is not a prediction. The existence of the intermediate band, its two boundaries a₁²/4 and a₀²/4, the locus Q∗(κ) and the ordering of the fold and the escape follow from the geometry and are reported as results. The absolute waiting time does not: it inherits the evidence rate and the regularisation of ref. 4, and varies by more than two orders of magnitude between the two selector scales used there. No clinical timescale is claimed from it.'),
 ('blank', ''),
 ('h1', 'Supplementary Note 4. Unequal coupling: the weakest stage'),
 ('p', 'Let each stage carry its own coupling, drₖ/dt = g(rₖ,Q) + κₖ(rₖ₋₁ − rₖ), and hold the threshold fixed at a. Two facts follow, and together they replace the equal-coupling assumption used in the main analyses.'),
 ('h2', 'S6. Propagation is decided stage by stage, by the weakest coupling'),
 ('p', 'Suppose the crossing has occurred and r₀ = 1. The first stage sees the equilibrium condition r₁(1 − r₁)(r₁ − a) + κ₁(1 − r₁) = (1 − r₁)[r₁² − a r₁ + κ₁] = 0, whose low roots exist exactly while κ₁ ≤ a²/4, so for κ₁ > a²/4 the first stage has no rest point below a and converges to 1. Stage 2 then faces a predecessor tending to 1 rather than one held at 1, and the step from that limit to a crossing in finite time is the content of the following lemma. Writing it out is what makes the induction a proof rather than a sequence of limits.'),
 ('h2', 'S6a. A predecessor tending to one carries a supercritical stage across'),
 ('p', 'Let stage k obey drₖ/dt = rₖ(1 − rₖ)(rₖ − a) + κₖ(s(t) − rₖ) with s(t) → 1, and let κₖ > a²/4. Fix θ with a < θ < 1 and put'),
 ('eq', '*μ_{θ} = min_{0 ≤ x ≤ θ} (1 − x)(x² − a x + κₖ).*'),
 ('p', 'Because the discriminant a² − 4κₖ is negative, x² − a x + κₖ ≥ κₖ − a²/4 > 0 for every real x, so μ_{θ} ≥ (1 − θ)(κₖ − a²/4) > 0. The field is affine and increasing in s with slope κₖ, so for s ≥ 1 − η it is bounded below on [0,θ] by μ_{θ} − κₖ η, which is positive for every η < μ_{θ}/κₖ. Choose such an η, and choose T with s(t) > 1 − η for all t > T. Then drₖ/dt ≥ μ_{θ} − κₖ η > 0 for every t > T at which rₖ ≤ θ, so rₖ leaves [0,θ] no later than'),
 ('eq', '*t = T + θ / (μ_{θ} − κₖ η),*'),
 ('p', 'which is finite. In particular rₖ crosses a in finite time. Above θ the same comparison against the field at s = 1 − η, whose only zero in (a,1] lies within O(η) of 1, gives rₖ(t) ≥ 1 − O(η) eventually; since η may be taken as small as desired once T is enlarged accordingly, rₖ → 1. Stage k therefore reproduces for stage k + 1 the hypothesis that stage k − 1 supplied for it, and the induction runs down the chain.'),
 ('p', 'The lemma also shows what the margin costs. At κₖ = 0.0901, a margin of 10^{−4} above a₀²/4 = 0.09, the field on [0,0.9] is bounded below by 7.0 × 10^{−5} and η must be smaller than 7.8 × 10^{−4}; at κₖ = 0.20 the bound is 4.7 × 10^{−2} and η may be as large as 0.52. Integrating the comparison equation at η = μ_{θ}/2κₖ gave crossing times of 627, 82, 29 and 9.3 at κₖ = 0.0901, 0.095, 0.12 and 0.20, in every case inside the bound above. A predecessor that merely converges, rather than one held at 1, therefore delays the crossing but cannot prevent it.'),
 ('p', 'Conversely, if κₖ < a²/4 the limiting field retains its low root at [a − √(a² − 4κₖ)]/2, the interval below it is forward invariant in the limit, and stage k converges there instead. Recovery therefore passes stage k if and only if κₖ > a²/4 and arrests at the first stage for which it does not, so the chain recovers completely if and only if minₖ κₖ > a²/4: the sharp quantity is the smallest coupling and not the average, and the arrest depth identifies which stage is weak. Numerically, bisecting each coupling separately with the others held at 1.0 returned the same boundary at every stage, 0.090000022 against a₀²/4 = 0.090000000, and the arrest depth matched the index of the first subcritical stage in every coupling vector tested, including vectors whose minimum sits within 0.001 of the boundary.'),
 ('h2', 'S6b. The intermediate band is stage-wise, at the running maximum of Q*'),
 ('p', "The same argument applies while the threshold is still moving. On the intermediate band a stage is held below its own unstable root until a(Q) falls to 2√κₖ, that is until Q reaches Q∗(κₖ). A stage cannot cross before its predecessor, so the level at which stage k completes is not Q∗(κₖ) but the running maximum of Q∗ over stages 1 to k: where a stage's own locus lies below that maximum it is released by its predecessor rather than by its own criterion, and it follows almost at once."),
 ('p', 'Integrating the full five-dimensional system with a coupling per stage, from the collapsed profile at Q = 0, at the five vectors (0.5, 0.05, 0.02), (0.5, 0.02, 0.05), (0.2, 0.08, 0.03), (0.3, 0.04, 0.07) and (0.5, 0.06, 0.015), all fifteen completions occurred in chain order and at evidence levels above the running maximum, by between 0 and 0.041. The overshoot is positive because Q continues to rise during the finite crossing, and it is larger for the stages released by a predecessor, which are already past their own locus when they go. Delayed completion is therefore not one event but a sequence of them, at levels the coupling vector fixes in advance.'),
 ('p', 'The intervals between consecutive completions are compressed relative to the wait that precedes them, from 0.004% to 9% across the five vectors, because a stage coming online multiplies the joint availability C(r) and accelerates Q. On the evidence coordinate the sequence is a clean staircase; in time it is often a burst. That asymmetry is itself a signature: late completions should not be evenly spaced.'),
 ('h2', 'S7. Existence of the branch is decided by the first stage alone'),
 ('p', 'For k ≥ 2 write f(x) = x(1 − x)(x − a) + κₖ(rₖ₋₁ − x) with 0 < rₖ₋₁ < a. Then f(0) = κₖ rₖ₋₁ > 0 and f(a) = −κₖ(a − rₖ₋₁) < 0, so f changes sign on (0,a) and a low root exists for every positive κₖ. No downstream coupling can destroy the branch; only κ₁ can, through the discriminant of the quadratic above.'),
 ('p', 'Two consequences follow. The critical coupling κ_{c} = a²/4 is a statement about the first stage and is therefore unchanged by the downstream couplings: the branch was present at κ₁ = 0.005625 and absent at 0.00563 for downstream couplings of 10^{−4}, 0.5, 10 and 30. It is also unchanged by the length of the chain, since adding stages adds only cubics that always have a low root: bisection returned 0.005625000000 at every chain length from one stage to twelve, agreeing to twelve digits. The identification of the coupling from the first stage ratio is untouched by both, because it uses only that stage.'),
 ('blank', ''),
 ('h1', 'Supplementary Note 5. The spectrum on the branch, and the exponent'),
 ('p', 'On a forward chain, stage k receives only from stage k − 1, so the Jacobian of the capacity block at frozen Q is lower triangular and its spectrum is its diagonal,'),
 ('eq', '*λₖ = (1 − 2rₖ)(rₖ − a) + rₖ(1 − rₖ) − κ.*'),
 ('p', 'Evaluate this at the first stage of the branch. There r₁ is the low root of r² − a r + κ = 0, so κ = a r₁ − r₁², and substituting gives λ₁ = (1 − 2r₁)(r₁ − a) + r₁(1 − r₁) − a r₁ + r₁² = 2r₁ − a − 2r₁² + a r₁. Writing D = √(a² − 4κ), the low root is r₁ = (a − D)/2, so 2r₁ − a = −D and −2r₁² + a r₁ = r₁(a − 2r₁) = r₁ D. Hence'),
 ('eq', '*λ₁ = −D (1 − r₁) = −√(a² − 4κ) (1 − r₁),*'),
 ('p', 'an identity, not an approximation. It was checked against the numerically differentiated diagonal at three thresholds and ten couplings per threshold, with a largest discrepancy of 1.1 × 10^{−16}.'),
 ('p', 'Three things follow. First, λ₁ < 0 wherever the branch exists, since D > 0 and r₁ < 1, and the downstream λₖ are the slopes of the cubics at their own low roots and are negative for the same reason, so the partial configuration is stable at every point of the branch and not only at the illustrative coupling. Second, the branch is normally hyperbolic away from the fold, so at a small evidence rate the full trajectory tracks it to first order in ε by the standard geometric singular perturbation estimate (ref. 22 of the main text); what that estimate does not cover is the passage through the fold itself. Third, λ₁ is the eigenvalue that vanishes at the fold, and it does so as the square root of the same discriminant whose vanishing defines κ_{c}. On the grid computed here, thirty combinations of threshold and coupling, λ₁ is also the least negative of the three, with a smallest margin of 8.8 × 10^{−3} against λ₂, but we use only the first statement, which is what the escape-time law requires. The critical coupling is the statement that the discriminant reaches zero; the escape exponent of one half is the statement that it reaches zero linearly in the excess, so that the rate, and hence the reciprocal of the escape time, vanishes as its square root. The two headline results are one expression read twice.'),
 ('blank', ''),
 ('h1', 'Supplementary Note 6. What the confinement statement does not cover'),
 ('p', 'The maximum principle of the main text has two hypotheses that are easy to read past: the coupling enters as a difference between capacities, and the threshold is not itself an intervention target. Three numerical experiments fix what each hypothesis is doing. All three use the illustrative parameters of ref. 4 and a chain of four capacities, and none of them contradicts the theorem; they delimit it.'),
 ('h2', 'S8. Additive excitatory coupling converts a sub-threshold chain on its own'),
 ('p', 'Replace the diffusive term by a non-negative excitatory drive on the complete graph,'),
 ('eq', '*drₖ/dt = rₖ(1 − rₖ)(rₖ − a) + w (1 − rₖ) Σ_{j ≠ k} S(rⱼ),   w ≥ 0,  S ≥ 0.*'),
 ('p', 'The gate (1 − rₖ) keeps the state in [0,1] and the weights are non-negative, so this is a legitimate excitatory coupling; what it lacks is the difference structure. The sign argument fails immediately: at an index attaining the maximum the drive is still strictly positive, so D⁺M is not bounded above by the uncoupled field and nothing prevents M from rising through a.'),
 ('p', 'It does rise. Started from rₖ = 0.30 for every k, wholly below a₀ = 0.60, with no external input and no change of threshold, the chain converts completely above w = 0.100 for S(r) = r and above w = 0.060 for the sigmoidal S(r) = [1 + exp(−(r − 0.30)/0.05)]^{−1}. Both weights are exact rather than fitted. For S(r) = r the symmetric subspace rₖ ≡ r carries the field r(1 − r)(r − a + 3w), so the additive drive shifts the threshold to a − 3w and the start at 0.30 crosses precisely when 3w > a₀ − 0.30, that is w > 0.100; bisection returned 0.100000000000. For the sigmoidal S the drift at the symmetric start is 0.30 × 0.70 × (0.30 − 0.60) + 3w × 0.70 × S(0.30) = −0.063 + 1.05 w, since S(0.30) = 1/2, and this changes sign at w = 0.060; bisection returned 0.059999999999.'),
 ('p', 'The diffusive control is unambiguous. The same start, the same graph and the same integration with the drive replaced by Σⱼ w (rⱼ − rₖ) leaves the chain at the failed corner at w = 0.1, 1, 10 and 100, three orders of magnitude above the additive threshold and far above any coupling used anywhere else in this work. The theorem is not a statement about coupling strength; it is a statement about coupling form, and no strength defeats it while no capacity is above threshold.'),
 ('p', 'The counterexample is not idle. The only biological observation this work has appealed to, an increase in co-firing within cell assemblies under stimulation (ref. 15 of the main text), describes an added excitatory drive rather than a redistribution along a difference, and so sits on the additive side of exactly this boundary. The two forms are not distinguished by anything measured here, and they make opposite predictions about the acute state: under diffusive coupling a wholly sub-threshold chain cannot be converted by coupling at any strength, so an acute conversion requires an excursion, whereas under an additive drive a large enough coupling change converts it with no excursion at all. The first prediction of the main text therefore separates the coupling forms as well as the two axes. If acute conversion is all-or-none in the delivered amplitude, with no group whose recovery is graded in amplitude, the excursion is doing the work and the diffusive reading survives; if response instead grades with the number or intensity of contacts independently of the amplitude reached, the additive reading is favoured and the confinement theorem does not apply to this system. That test uses dose-response data of a kind that already exists.'),
 ('h2', 'S9. Lowering the threshold converts the chain with no input and no coupling change'),
 ('p', 'The main text takes three quantities to be externally accessible. That list is a modelling assumption, and the threshold parameters show what it excludes. Holding the coupling at the reference value 0.6 and supplying no input at all, a chain started at 0.30 everywhere fails at a = 0.60 and at a = 0.40, sits exactly on its unstable fixed point at a = 0.30, and recovers completely at a = 0.25. Nothing in the theorem is violated, because a is a parameter of the field rather than a variable of the dynamics, but an intervention that lowered the prevailing threshold would convert a collapsed chain without any excursion and without touching the coupling. Any claim that an acute conversion requires an excursion is conditional on the threshold being fixed, and that conditionality is stated in the main text rather than assumed away.'),
 ('h2', 'S10. A crossing at the root propagates only under supercritical coupling'),
 ('p', 'The pulse criterion of the main text, that the chain converts if and only if the root ends the pulse above a₀, is a statement about the root together with a coupling that carries the rest of the chain. Placing the root just above threshold at r₀ = 0.61 with the remaining capacities at zero and integrating to 3 × 10^{4}, the whole chain recovers at κ = 0.6, 0.2 and 0.095, and does not at κ = 0.085, 0.05 and 0.01. At κ = 0.085 the state settles at (1, 0.2293, 0.0306, 0.0038), the root recovered and the chain arrested on the partial branch. The boundary is a₀²/4 = 0.090, the same criterion as everywhere else in this work, so the two statements are consistent; the pulse sweep of Fig. 1c was run at κ = 0.6, which is supercritical, and the equivalence there is between the pulse and the crossing of the root, not between the pulse and recovery of the chain at arbitrary coupling.'),
]



# --------------------------------------------------------------------------
# Word output
# --------------------------------------------------------------------------
RPR_BASE = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
            'w:cs="Times New Roman"/>')
SZ = '<w:sz w:val="24"/><w:szCs w:val="24"/>'
TOKEN = re.compile(r'\^\{(.+?)\}|_\{(.+?)\}|\*\*(.+?)\*\*|\*(.+?)\*')
NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS_PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
EMU_PER_IN = 914400
TEXT_WIDTH_IN = (11906 - 1701 - 1701) / 1440.0      # A4 minus the section margins


def _esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _runs(text, bold=False):
    out, pos = [], 0

    def emit(t, sup=False, sub=False, ital=False, bld=bold):
        if not t:
            return
        rpr = '<w:rPr>' + RPR_BASE
        if bld:
            rpr += '<w:b/><w:bCs/>'
        if ital:
            rpr += '<w:i/><w:iCs/>'
        rpr += SZ
        if sup:
            rpr += '<w:vertAlign w:val="superscript"/>'
        if sub:
            rpr += '<w:vertAlign w:val="subscript"/>'
        rpr += '</w:rPr>'
        out.append('<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr, _esc(t)))

    for m in TOKEN.finditer(text):
        emit(text[pos:m.start()])
        if m.group(1) is not None:
            emit(m.group(1), sup=True)
        elif m.group(2) is not None:
            emit(m.group(2), sub=True)
        elif m.group(3) is not None:
            emit(m.group(3), bld=True)
        else:
            emit(m.group(4), ital=True)
        pos = m.end()
    emit(text[pos:])
    return ''.join(out)


def _png_size(path):
    with open(path, 'rb') as fh:
        head = fh.read(24)
    return struct.unpack('>II', head[16:24])


def _drawing(path, rid, idx):
    w, h = _png_size(path)
    cx = int(TEXT_WIDTH_IN * EMU_PER_IN)
    cy = int(cx * h / w)
    return (
        '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        '<wp:extent cx="%d" cy="%d"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<wp:docPr id="%d" name="Figure %d"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="%s" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="%s"><a:graphicData uri="%s">'
        '<pic:pic xmlns:pic="%s">'
        '<pic:nvPicPr><pic:cNvPr id="%d" name="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
        '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        % (cx, cy, 1000 + idx, idx, NS_A, NS_A, NS_PIC, NS_PIC,
           1000 + idx, os.path.basename(path), rid, cx, cy))


def build_docx(paras, template, out_path):
    """Render a paragraph list into a .docx, reusing the template's page setup."""
    import shutil
    import tempfile
    import zipfile

    figures = []

    def para(kind, text):
        centre = '<w:jc w:val="center"/>' if kind in ('title', 'eq', 'fig') else ''
        bold = kind in ('title', 'h1')
        spacing = ('<w:spacing w:line="240" w:lineRule="auto" w:before="120" w:after="120"/>'
                   if kind in ('fig', 'leg') else
                   '<w:spacing w:line="480" w:lineRule="auto"/>')
        mark = '<w:rPr>' + RPR_BASE + ('<w:b/><w:bCs/>' if bold else '') + SZ + '</w:rPr>'
        ppr = '<w:pPr>' + spacing + centre + mark + '</w:pPr>'
        if kind == 'blank':
            inner = ''
        elif kind == 'tab':
            inner = '<w:r><w:rPr>' + RPR_BASE + SZ + '</w:rPr><w:tab/></w:r>'
        elif kind == 'pagebreak':
            inner = '<w:r><w:rPr>' + RPR_BASE + SZ + '</w:rPr><w:br w:type="page"/></w:r>'
        elif kind == 'fig':
            idx = len(figures) + 1
            rid = 'rIdFig%d' % idx
            figures.append((text, rid))
            inner = _drawing(text, rid, idx)
        else:
            inner = _runs(text, bold=bold)
        return '<w:p>' + ppr + inner + '</w:p>'

    body = ''.join(para(k, t) for k, t in paras)
    work = tempfile.mkdtemp(prefix='docx_')
    try:
        with zipfile.ZipFile(template) as z:
            z.extractall(work)
        for root, _, files in os.walk(work):
            for fn in files:
                p = os.path.join(root, fn)
                if os.path.islink(p):
                    os.unlink(p)
        doc = os.path.join(work, 'word', 'document.xml')
        src = open(doc, encoding='utf-8').read()
        head = src[:src.index('<w:body>') + len('<w:body>')]
        sectpr = re.search(r'<w:sectPr.*?</w:sectPr>', src, re.S).group()
        open(doc, 'w', encoding='utf-8').write(head + body + sectpr + '</w:body></w:document>')

        media = os.path.join(work, 'word', 'media')
        shutil.rmtree(media, ignore_errors=True)
        rels_path = os.path.join(work, 'word', '_rels', 'document.xml.rels')
        rels = open(rels_path, encoding='utf-8').read()
        rels = re.sub(r'<Relationship [^>]*Target="media/[^"]*"[^>]*/>', '', rels)
        if figures:
            os.makedirs(media, exist_ok=True)
            extra = ''
            for fname, rid in figures:
                shutil.copy(fname, os.path.join(media, os.path.basename(fname)))
                extra += ('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
                          'officeDocument/2006/relationships/image" Target="media/%s"/>'
                          % (rid, os.path.basename(fname)))
            rels = rels.replace('</Relationships>', extra + '</Relationships>')
        open(rels_path, 'w', encoding='utf-8').write(rels)

        if os.path.exists(out_path):
            os.remove(out_path)
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(work):
                for fn in files:
                    p = os.path.join(root, fn)
                    z.write(p, os.path.relpath(p, work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path


def build_documents(template):
    """Write the manuscript and the Supplementary Information."""
    missing = [f for f in ('Fig1.png', 'Fig2.png', 'Fig3.png') if not os.path.exists(f)]
    if missing:
        raise SystemExit('missing %s; run the analysis or --figures first'
                         % ', '.join(missing))
    a = build_docx(MANUSCRIPT_PARAS, template, 'Saito_two_axes_NatComms.docx')
    b = build_docx(SUPPLEMENT_PARAS, template, 'Saito_two_axes_SupplementaryInformation.docx')
    print('wrote %s and %s' % (a, b))


# --------------------------------------------------------------------------
# Nature Communications submission compliance audit
# --------------------------------------------------------------------------
def compliance_audit(verbose=True):
    """Check the manuscript against the formal Nature Communications rules.

    Article: title <= 15 words and free of punctuation; abstract <= 150 words
    with no references and no abbreviations; main text (Introduction through
    Conclusion, excluding abstract, Methods, references and figure legends)
    <= 5000 words; unnumbered subheadings <= 60 characters; at most 10 display
    items; references numbered in order of first appearance in the text;
    figures and their panels cited in order; no em dashes anywhere.

    Returns (ok, rows) with rows = [(name, detail, passed), ...].
    """
    P, S = MANUSCRIPT_PARAS, SUPPLEMENT_PARAS
    text_kinds = ('p', 'eq', 'h1', 'h2')

    def words(t):
        return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]*", t))

    def section(a, b):
        i = next(i for i, q in enumerate(P) if q[0] == 'h1' and q[1] == a)
        j = next(i for i, q in enumerate(P) if q[0] == 'h1' and q[1] == b)
        return ' '.join(q[1] for q in P[i:j] if q[0] in text_kinds)

    rows = []

    def add(name, ok, detail):
        rows.append((name, detail, bool(ok)))

    title = P[0][1]
    add('title <= 15 words', len(title.split()) <= 15, '%d words' % len(title.split()))
    add('title free of punctuation', not re.search(r'[.,;:!?]', title), repr(title))

    abstract = section('Abstract', 'Introduction')
    abstract = ' '.join(q[1] for q in P if q[0] == 'p'
                        and P.index(q) > [i for i, x in enumerate(P)
                                          if x[0] == 'h1' and x[1] == 'Abstract'][0]
                        and P.index(q) < [i for i, x in enumerate(P)
                                          if x[0] == 'h1' and x[1] == 'Introduction'][0])
    add('abstract <= 150 words', words(abstract) <= 150, '%d words' % words(abstract))
    add('abstract cites no references', '^{' not in abstract, 'no superscript citations')

    body = section('Introduction', 'Methods')
    n_alpha, n_raw = words(body), len(body.split())
    add('main text <= 5000 words', max(n_alpha, n_raw) <= 5000,
        '%d words / %d whitespace tokens' % (n_alpha, n_raw))

    longs = [q[1] for q in P if q[0] == 'h2' and len(q[1]) > 60]
    add('subheadings <= 60 characters', not longs, longs or 'longest %d chars'
        % max([len(q[1]) for q in P if q[0] == 'h2'] or [0]))

    nfig = sum(1 for q in P if q[0] == 'fig')
    add('display items <= 10', nfig <= 10, '%d figures' % nfig)

    em = sum(q[1].count('\u2014') for q in P) + sum(q[1].count('\u2014') for q in S)
    add('no em dashes', em == 0, '%d found' % em)

    iref = next(i for i, q in enumerate(P) if q[0] == 'h1' and q[1] == 'References')
    cited_text = ' '.join(q[1] for q in P[:iref]
                          if q[0] in text_kinds + ('leg', 'title'))
    order = []
    for m in re.finditer(r'\^\{([0-9,\s\-\u2013]+)\}', cited_text):
        for tok in m.group(1).split(','):
            tok = tok.strip()
            if re.search(r'[-\u2013]', tok):
                lo, hi = re.split(r'[-\u2013]', tok)
                order += list(range(int(lo), int(hi) + 1))
            elif tok:
                order.append(int(tok))
    first = []
    for n in order:
        if n not in first:
            first.append(n)
    listed = [q[1] for q in P[iref + 1:] if q[0] == 'p' and re.match(r'^\d+\.', q[1])]
    add('references in first-appearance order', first == sorted(first),
        'first appearance %s' % (first if first != sorted(first) else 'ascending'))
    add('every listed reference is cited', set(first) == set(range(1, len(listed) + 1)),
        '%d listed, %d cited' % (len(listed), len(set(first))))
    # a reference is locatable if it carries a DOI or URL, a bold volume number,
    # or a publisher and year, which is how books are cited.
    noid = [r for r in listed
            if not re.search(r'https?://|doi\.org|\*\*\d|\([A-Z][^()]*, \d{4}\)', r)]
    add('every reference is locatable', not noid,
        noid[0][:70] + ' ...' if noid else 'DOI, URL, volume or publisher')
    stub = [q[1] for q in P
            if re.search(r'\[[^\]]*(to be inserted|GitHub user)[^\]]*\]', q[1])]
    add('no placeholder identifiers remain', not stub,
        '%d passage%s awaiting an identifier' % (len(stub), '' if len(stub) == 1 else 's')
        if stub else 'none')

    seen = []
    for m in re.finditer(r'Fig\.\s*(\d)([a-z])?', cited_text):
        key = (int(m.group(1)), m.group(2) or '')
        if key not in seen:
            seen.append(key)
    fignums = []
    for f, _ in seen:
        if f not in fignums:
            fignums.append(f)
    add('figures cited in numerical order', fignums == sorted(fignums), str(fignums))
    panel_ok, bad = True, []
    for f in sorted(set(fignums)):
        p = [q for n, q in seen if n == f and q]
        if p != sorted(p):
            panel_ok, bad = False, bad + ['Fig. %d %s' % (f, p)]
    add('panels cited in order within each figure', panel_ok, bad or 'a, b, c, ... in each')

    have = sorted(set(int(x) for x in
                      re.findall(r'Supplementary Note (\d)\.',
                                 ' '.join(q[1] for q in S))))
    want = set(int(x) for x in
               re.findall(r'Supplementary Notes? (\d)', ' '.join(q[1] for q in P)))
    for a, b in re.findall(r'Supplementary Notes (\d) and (\d)',
                           ' '.join(q[1] for q in P)):
        want |= {int(a), int(b)}
    add('every cited Supplementary Note exists', want <= set(have),
        'cited %s, present %s' % (sorted(want), have))

    order_ok = [q[1] for q in P if q[0] == 'h1']
    expected = ['Abstract', 'Introduction', 'Results', 'Discussion']
    add('section order follows the Article format',
        order_ok[:4] == expected and 'Methods' in order_ok
        and order_ok.index('Methods') > order_ok.index('Discussion'),
        ' > '.join(order_ok[:6]))

    if verbose:
        print('Nature Communications compliance')
        print('-' * 76)
        for name, detail, ok in rows:
            print('%-44s %-24s %s' % (name[:44], str(detail)[:24],
                                      'ok' if ok else 'CHECK'))
        print('-' * 76)
        nbad = sum(1 for _, _, ok in rows if not ok)
        print('%d checks, %d needing attention' % (len(rows), nbad))
    return all(ok for _, _, ok in rows), rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='coarser grids, ~3 min')
    ap.add_argument('--figures', action='store_true',
                    help='regenerate figures from an existing results.json')
    ap.add_argument('--verify', action='store_true',
                    help='check every value the manuscript reports against results.json')
    ap.add_argument('--docx', metavar='TEMPLATE',
                    help='write the manuscript and Supplementary Information, reusing '
                         'the page setup and styles of TEMPLATE (a .docx); needs Fig1-4.png')
    ap.add_argument('--audit', action='store_true',
                    help='check the manuscript against the Nature Communications rules')
    ap.add_argument('--outdir', default=None,
                    help='directory for results.json, the figures and the documents '
                         '(created if absent; defaults to the working directory)')
    if argv is None:
        # Jupyter and Colab append their own arguments (-f kernel.json) to
        # sys.argv, so ignore the command line entirely inside a kernel.
        argv = [] if in_notebook() else sys.argv[1:]
    args = ap.parse_args(argv)

    template = os.path.abspath(args.docx) if args.docx else None
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        os.chdir(args.outdir)
    if not os.access('.', os.W_OK):
        ap.error('the output directory is not writable; pass --outdir <dir>')

    # The steps compose: any combination of flags runs the ones asked for, in
    # the order analysis -> figures -> verify -> documents, and whatever a later
    # step needs is produced first if it is missing.
    if args.audit and not (args.figures or args.verify or template):
        return 0 if compliance_audit()[0] else 1

    reuse = (args.figures or args.verify or template) and os.path.exists('results.json')
    if reuse:
        D = json.load(open('results.json'))
        print('read results.json')
    else:
        D = run_all(quick=args.quick)
        with open('results.json', 'w') as fh:
            json.dump(D, fh, indent=1)
        print('wrote results.json')

    plain = not (args.figures or args.verify or template or args.audit)
    if plain or args.figures or (template and not all(
            os.path.exists(f) for f in ('Fig1.png', 'Fig2.png', 'Fig3.png', 'Fig4.png'))):
        make_figures(D)
        print('wrote Fig1 to Fig4 (pdf and png)')
        write_source_data(D)

    status = 0
    if args.verify:
        status = 0 if verify(D) else 1
    if args.audit:
        print()
        if not compliance_audit()[0]:
            status = status or 1
    if template:
        build_documents(template)
    if plain or args.figures:
        show_figures()
    return status


if __name__ == '__main__':
    _code = main()
    if in_notebook():
        if _code:
            print('finished with status %d' % _code)
    else:
        sys.exit(_code)
