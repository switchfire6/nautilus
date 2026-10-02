"""Rung 1: simulate batches of Löwe gated networks and classify insight."""

from __future__ import annotations

import numpy as np

from .learners import G_C, G_M, W_C, W_M, GatedConfig, analytic_accuracy, train_lowe
from .measures import binned_mean, classify_insight, fit_switch, switch_delay_trials
from .tasks import SSSTConfig, make_ssst_trials

BIN = 50  # trials per bin ("halved task blocks")


def simulate(task: SSSTConfig, net: GatedConfig, mu_table: np.ndarray, rng) -> dict:
    """Train one network per row of ``mu_table`` through the whole curriculum.

    Returns trials, parameter history (N, T+1, 4), data gradients (N, T, 4),
    per-trial analytic accuracy (N, T) and the hardest-coherence accuracy in 50-trial
    bins over the motion and colour phases (N, 12 with the default curriculum).
    """
    trials = make_ssst_trials(task, mu_table, rng)
    theta0 = np.full((len(mu_table), 4), net.init)
    hist, grads = train_lowe(theta0, trials, net, rng)
    acc = analytic_accuracy(hist[:, :-1], trials.mu_m, trials.mu_c,
                            task.sigma_m, task.sigma_c, net.sigma_eta)
    tp = task.n_pretrain
    series = binned_mean(acc[:, tp:], trials.coh[:, tp:] == 0, BIN)
    return dict(trials=trials, hist=hist, grads=grads, acc=acc, series=series)


def fit_all(series: np.ndarray, n_pre: int, method: str = "cobyla") -> dict:
    n_bins = series.shape[1]
    bounds = ((0.35, 1.0), (0.0, n_bins), (0.0, 9.0))
    fits = [fit_switch(s, n_pre=n_pre, bounds=bounds, method=method) for s in series]
    return {k: np.array([getattr(f, k) for f in fits])
            for k in ("fmin", "fmax", "t_switch", "slope", "sse", "steepness", "score")}


def replicate_batch(
    task: SSSTConfig,
    net: GatedConfig,
    mu_table: np.ndarray,
    rng,
    fit_methods: tuple[str, ...] = ("cobyla", "global"),
) -> dict:
    """One replication batch: N test networks plus N control networks (colour never
    predictive), classified as in Löwe et al. (2024) with each fit method.
    Returns summary statistics keyed by fit method."""
    test = simulate(task, net, mu_table, rng)
    ctrl = simulate(task.as_control(), net, mu_table, rng)
    return {m: classify_batch(task, test, ctrl, m) for m in fit_methods}


def classify_batch(task: SSSTConfig, test: dict, ctrl: dict, fit_method: str) -> dict:
    n_pre = task.n_motion // BIN
    ft = fit_all(test["series"], n_pre, fit_method)
    fc = fit_all(ctrl["series"], n_pre, fit_method)
    insight = classify_insight(ft["score"], fc["score"])
    delay = switch_delay_trials(ft["t_switch"][insight], n_pre, BIN)

    on = task.colour_onset
    theta_on = test["hist"][:, on]
    theta_end = test["hist"][:, -1]
    # "silent knowledge": colour-gate gradient magnitude in the first 5 colour trials
    gc_grad = np.abs(test["grads"][:, on : on + 5, G_C]).mean(axis=1)

    def split(x):
        return dict(insight=_ms(x[insight]), no_insight=_ms(x[~insight]))

    last_block = test["acc"][:, -100:].mean(axis=1)
    motion_low = test["series"][:, :n_pre].mean(axis=1)
    colour_low = test["series"][:, n_pre:].mean(axis=1)
    return dict(
        n=len(insight),
        n_insight=int(insight.sum()),
        frac_insight=float(insight.mean()),
        control_max_score=float(fc["score"].max()),
        delay_trials=_ms(delay),
        t_switch_bins=_ms(ft["t_switch"][insight]),
        slope_fitted=split(ft["slope"]),
        gc_grad_first5=split(gc_grad),
        abs_wc_onset=split(np.abs(theta_on[:, W_C])),
        abs_wm_onset=split(np.abs(theta_on[:, W_M])),
        abs_gc_onset=split(np.abs(theta_on[:, G_C])),
        abs_gm_onset=split(np.abs(theta_on[:, G_M])),
        abs_gc_end=split(np.abs(theta_end[:, G_C])),
        abs_gm_end=split(np.abs(theta_end[:, G_M])),
        acc_motion_phase_low=split(motion_low),
        acc_colour_phase_low=split(colour_low),
        acc_last_block=_ms(last_block),
        insight_mask=insight.astype(int).tolist(),
        scores=ft["score"].tolist(),
        control_scores=fc["score"].tolist(),
        t_switch=ft["t_switch"].tolist(),
        slopes=ft["slope"].tolist(),
    )


def _ms(x) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), n=0)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                n=int(x.size))
