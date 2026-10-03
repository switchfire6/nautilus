"""Measures: behavioural insight, internal compression, and lead time.

Insight follows Löwe et al. (2024) and their analysis script (SwitchNetsBehavior.R):
a 3-parameter sigmoid is fitted by COBYLA to binned accuracy on the hardest trials,
its slope at the inflection point is corrected by subtracting the fit's squared error,
and a learner counts as an insight learner if this score exceeds the *maximum* score
of control learners that never saw the hidden regularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import nlopt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def binned_mean(values: np.ndarray, mask: np.ndarray, bin_size: int) -> np.ndarray:
    """Mean of ``values`` over trials where ``mask`` is true, in consecutive bins.

    ``values`` and ``mask`` have shape (..., T); T must be a multiple of ``bin_size``.
    Bins without masked trials give NaN.
    """
    t = values.shape[-1]
    if t % bin_size:
        raise ValueError("T must be a multiple of bin_size")
    shape = values.shape[:-1] + (t // bin_size, bin_size)
    v = np.where(mask, values, 0.0).reshape(shape)
    m = np.broadcast_to(mask, values.shape).reshape(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v.sum(-1) / m.sum(-1)


def sigmoid(t, fmin, fmax, slope, t_switch):
    return (fmax - fmin) / (1.0 + np.exp(-slope * (t - t_switch))) + fmin


@dataclass(frozen=True)
class SwitchFit:
    fmin: float  # mean of the pre-change bins (fixed, not fitted)
    fmax: float
    t_switch: float  # inflection point, in bins (1-based time axis as in the R script)
    slope: float  # sigmoid steepness parameter m
    sse: float
    steepness: float  # slope at the inflection point: m * (fmax - fmin) / 4
    score: float  # steepness - sse (the corrected steepness used for classification)


def fit_switch(
    series: np.ndarray,
    n_pre: int = 4,
    x0: tuple[float, float, float] = (0.5, 4.0, 7.0),
    bounds: tuple = ((0.35, 1.0), (0.0, 12.0), (0.0, 9.0)),
    method: str = "cobyla",
) -> SwitchFit:
    """Fit the Löwe et al. sigmoid to one learner's binned accuracy series.

    Free parameters are (fmax, t_switch, slope); time runs 1..len(series).

    - ``cobyla``: one local NLopt COBYLA run from ``x0`` = (0.5, 4, 7) with
      ``xtol_rel`` = 1e-8 and at most 10^4 evaluations: the authors' exact call (R nloptr
      wraps the same NLopt library). It can stop in a local optimum near the start.
    - ``global``: bounded L-BFGS-B from a grid of starts (plus ``x0``); returns the
      best least-squares fit found.
    """
    y = np.asarray(series, dtype=float)
    t = np.arange(1, len(y) + 1)
    fmin = float(np.mean(y[:n_pre]))

    def loss(p):
        return float(np.sum((y - sigmoid(t, fmin, p[0], p[2], p[1])) ** 2))

    if method == "cobyla":
        opt = nlopt.opt(nlopt.LN_COBYLA, 3)
        opt.set_lower_bounds([b[0] for b in bounds])
        opt.set_upper_bounds([b[1] for b in bounds])
        opt.set_min_objective(lambda p, grad: loss(p))
        opt.set_xtol_rel(1e-8)
        opt.set_maxeval(10_000)
        best = opt.optimize(np.asarray(x0, dtype=float))
    elif method == "global":
        (_, _), (t_lo, t_hi), (_, m_hi) = bounds
        starts = [np.asarray(x0)] + [
            np.array([max(y.max(), bounds[0][0]), ts, m])
            for ts in np.linspace(t_lo + 1, t_hi - 1, 4)
            for m in (0.5, 2.0, m_hi * 0.8)
        ]
        fits = [minimize(loss, s, method="L-BFGS-B", bounds=bounds) for s in starts]
        best = min(fits, key=lambda r: r.fun).x
    else:
        raise ValueError(method)
    fmax, ts, m = (float(v) for v in best)
    sse = loss(best)
    steep = m * (fmax - fmin) / 4.0
    return SwitchFit(fmin, fmax, ts, m, sse, steep, steep - sse)


def classify_insight(scores: np.ndarray, control_scores: np.ndarray) -> np.ndarray:
    """Insight if the score exceeds the maximum control score (100th percentile).

    Diverged learners (NaN scores) are ignored among controls and count as no insight.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(scores) > np.nanmax(control_scores)


def switch_delay_trials(t_switch, n_pre: int = 4, bin_size: int = 50):
    """Delay of the switch after the regularity onset, in trials."""
    return (np.asarray(t_switch) - n_pre) * bin_size


# ---------------------------------------------------------------------------
# Compression of a point cloud (e.g. hidden states) or of a parameter vector
# ---------------------------------------------------------------------------


def _unique_rows(x: np.ndarray) -> np.ndarray:
    return np.unique(np.round(np.asarray(x, dtype=float), 12), axis=0)


def twonn_dimension(x: np.ndarray, discard: float = 0.1) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017).

    Uses the ratio mu = r2/r1 of each point's two nearest-neighbour distances and the
    linear fit of -log(1 - F(mu)) against log(mu) through the origin, discarding the
    largest ``discard`` fraction of ratios. Duplicate points are removed first.
    """
    x = _unique_rows(x)
    if len(x) < 4:
        return float("nan")
    d = squareform(pdist(x))
    np.fill_diagonal(d, np.inf)
    two = np.partition(d, 1, axis=1)[:, :2]
    mu = np.sort(two[:, 1] / two[:, 0])
    n = len(mu)
    keep = int(np.floor(n * (1.0 - discard)))
    f = np.arange(1, n + 1) / n
    xs, ys = np.log(mu[:keep]), -np.log(1.0 - f[:keep])
    return float(np.dot(xs, ys) / np.dot(xs, xs))


def correlation_dimension(x: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.5) -> float:
    """Grassberger-Procaccia correlation dimension: the slope of log C(r) against log r,
    fitted between the ``q_lo`` and ``q_hi`` quantiles of pairwise distances."""
    x = _unique_rows(x)
    if len(x) < 4:
        return float("nan")
    dist = np.sort(pdist(x))
    r = np.geomspace(np.quantile(dist, q_lo), np.quantile(dist, q_hi), 12)
    c = np.searchsorted(dist, r, side="right") / len(dist)
    ok = c > 0
    return float(np.polyfit(np.log(r[ok]), np.log(c[ok]), 1)[0])


def effective_rank(x: np.ndarray, center: bool = True) -> float:
    """Roy & Vetterli (2007) effective rank: exp of the entropy of normalised singular
    values."""
    x = np.asarray(x, dtype=float)
    if center:
        x = x - x.mean(axis=0)
    s = np.linalg.svd(x, compute_uv=False)
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def gate_sparsity(g: np.ndarray, eps: float = 1e-3) -> float:
    """Fraction of gates with magnitude below ``eps``."""
    return float(np.mean(np.abs(np.asarray(g)) < eps))


def hoyer_sparsity(g: np.ndarray) -> float:
    """Hoyer (2004) sparsity in [0, 1]: 0 for equal magnitudes, 1 for a single non-zero."""
    g = np.abs(np.asarray(g, dtype=float)).ravel()
    n = g.size
    l2 = np.sqrt(np.sum(g**2))
    if l2 == 0 or n == 1:
        return 0.0
    return float((np.sqrt(n) - g.sum() / l2) / (np.sqrt(n) - 1))


# ---------------------------------------------------------------------------
# Change points and lead time
# ---------------------------------------------------------------------------


def change_point_mean(y: np.ndarray, min_seg: int = 2) -> int:
    """Index of the best single split of ``y`` into two constant segments (least squares).
    Returns the first index of the second segment."""
    y = np.asarray(y, dtype=float)
    best, best_k = np.inf, min_seg
    for k in range(min_seg, len(y) - min_seg + 1):
        a, b = y[:k], y[k:]
        sse = np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)
        if sse < best:
            best, best_k = sse, k
    return best_k


def onset_hinge(y: np.ndarray, min_pre: int = 2) -> int:
    """Onset of a change: the best ``k`` for a model that is constant before ``k`` and
    linear after it (continuous at ``k``). Returns ``k`` as an index into ``y``."""
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y), dtype=float)
    best, best_k = np.inf, min_pre
    for k in range(min_pre, len(y) - 1):
        design = np.stack([np.ones_like(t), np.maximum(t - k, 0.0)], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        sse = np.sum((design @ coef - y) ** 2)
        if sse < best:
            best, best_k = sse, k
    return best_k


def lead_time(compression_change: float, behavioural_switch: float) -> float:
    """Compression change point minus behavioural switch point (same time units).
    Negative values mean compression changed *before* behaviour."""
    return float(compression_change - behavioural_switch)
