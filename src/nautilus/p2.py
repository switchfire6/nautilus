"""Protocol P2: does early compression predict insight?

See docs/protocols/P2_compression_predicts.md.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from .harness import rng_for, seed_for
from .learners import NRTConfig, NRTLearner
from .measures import correlation_dimension, effective_rank, onset_hinge, twonn_dimension
from .p1 import _idx, _online
from .rung2 import NRTCurriculum, instruct
from .tasks import EARLY_STEP, make_nrt_strings

N_PRE, N_MIRROR, WINDOW, LAST = 2000, 5000, 1000, 400
GATE_L1 = 0.01
PREDICTORS = ("P_twonn", "P_corr", "P_rank", "K_silent", "K_move")


def probe_times() -> list[int]:
    return list(range(0, WINDOW, 100)) + list(range(WINDOW, N_MIRROR + 1, 250))


def _flat(net) -> torch.Tensor:
    return torch.cat([p.detach().ravel() for p in net.parameters()])


def run_learner(key: tuple) -> dict:
    cfg = NRTConfig(gate_l1=GATE_L1)
    rng = rng_for(*key, "session")
    learner = NRTLearner(cfg, seed_for(*key, "init"))
    instruct(learner, NRTCurriculum(), rng)
    d0, r0 = make_nrt_strings(N_PRE, rng, mirror=False)
    _online(learner, _idx(d0), _idx(r0), rng)
    dt, rt = make_nrt_strings(500, rng_for(*key, "heldout"), mirror=False)
    with torch.no_grad():
        _, rl, _ = learner.net(_idx(dt))
        step_acc = float((rl[:, 1:].argmax(-1) == _idx(rt)).float().mean())

    pd, pr = make_nrt_strings(300, rng_for(*key, "probe"), mirror=True)
    probe, answer = _idx(pd), _idx(pr)[:, -1]
    dm, rm = make_nrt_strings(N_MIRROR, rng, mirror=True)
    dm, rm = _idx(dm), _idx(rm)
    theta0 = _flat(learner.net)
    probes, pending = [], list(probe_times())
    ec = np.empty(N_MIRROR, dtype=bool)
    b = cfg.batch
    for i in range(0, N_MIRROR + b, b):
        i = min(i, N_MIRROR)
        if pending and i >= pending[0]:  # first update at or after each probe time
            while pending and i >= pending[0]:
                t_probe = pending.pop(0)
            with torch.no_grad():
                h, _, fl = learner.net(probe)
                p_ans = torch.softmax(fl[:, EARLY_STEP - 1], -1)
                p_correct = float(p_ans.gather(1, answer[:, None]).mean())
            hs = h[:, EARLY_STEP - 1].numpy().astype(float)
            probes.append(dict(trial=t_probe, twonn=twonn_dimension(hs),
                               corrdim=correlation_dimension(hs), erank=effective_rank(hs),
                               p_correct=p_correct,
                               move=float((_flat(learner.net) - theta0).norm())))
        if i == N_MIRROR:
            break
        ec[i : i + b] = _online(learner, dm[i : i + b], rm[i : i + b], rng)
    return dict(key=list(key), step_acc=step_acc, final=float(ec[-LAST:].mean()),
                bins=ec.reshape(-1, 250).mean(axis=1).tolist(), probes=probes)


def predictors(run: dict) -> dict[str, float]:
    pr = {p["trial"]: p for p in run["probes"]}
    a, b = pr[0], pr[WINDOW]
    return dict(P_twonn=-(b["twonn"] - a["twonn"]), P_corr=-(b["corrdim"] - a["corrdim"]),
                P_rank=-(b["erank"] - a["erank"]), K_silent=b["p_correct"] - a["p_correct"],
                K_move=b["move"])


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC; ties count one half."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def _logit_fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    xb = np.c_[np.ones(len(x)), x]

    def nll(w):
        z = xb @ w
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[1:] ** 2)

    def grad(w):
        p = 1 / (1 + np.exp(-(xb @ w)))
        g = xb.T @ (p - y)
        g[1:] += 2 * l2 * w[1:]
        return g

    return minimize(nll, np.zeros(xb.shape[1]), jac=grad, method="L-BFGS-B").x


def cv_auc(x: np.ndarray, y: np.ndarray, folds: int = 10, seed: int = 0) -> float:
    """Cross-validated AUC of a logistic regression (stratified folds, standardised)."""
    x, y = np.atleast_2d(np.asarray(x, float).T).T, np.asarray(y, float)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % folds
    pred = np.empty(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        if not te.any():
            continue
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-12
        w = _logit_fit((x[tr] - mu) / sd, y[tr])
        pred[te] = np.c_[np.ones(te.sum()), (x[te] - mu) / sd] @ w
    return auc(pred, y.astype(bool))


def bootstrap(fn, n: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def analyse(runs: list[dict], n_boot: int = 2000) -> dict:
    y = np.array([r["final"] > 0.5 for r in runs])
    pr = {k: np.array([predictors(r)[k] for r in runs]) for k in PREDICTORS}
    aucs = {k: auc(v, y) for k, v in pr.items()}
    n = len(y)
    c1_ci = bootstrap(lambda i: auc(pr["P_twonn"][i], y[i]), n, n_boot, 1)
    both = np.c_[pr["K_silent"], pr["P_twonn"]]

    def gain(i=None):
        i = np.arange(n) if i is None else i
        return cv_auc(both[i], y[i]) - cv_auc(pr["K_silent"][i], y[i])

    g = gain()
    g_ci = bootstrap(gain, n, n_boot, 2)
    # descriptive lead: TwoNN hinge onset minus first 250-string bin above 0.5
    leads = []
    for r, s in zip(runs, y):
        if not s:
            continue
        t = np.array([p["trial"] for p in r["probes"]])
        tw = np.array([p["twonn"] for p in r["probes"]])
        onset = t[onset_hinge(tw)]
        sw = (int(np.argmax(np.array(r["bins"]) > 0.5)) + 0.5) * 250
        leads.append(float(onset - sw))
    return dict(
        n=n, switch_share=float(y.mean()), aucs=aucs,
        C1=dict(auc=aucs["P_twonn"], ci=c1_ci,
                passed=bool(aucs["P_twonn"] >= 0.65 and c1_ci[0] > 0.5)),
        C2=dict(gain=g, ci=g_ci, passed=bool(g_ci[0] > 0)),
        lead_strings=dict(values=leads, median=float(np.median(leads)) if leads else None),
        step_acc_min=float(min(r["step_acc"] for r in runs)),
    )
