"""Protocol P1: sleep versus equal awake compute (docs/protocols/P1_sleep_vs_awake.md).

Paired design: each learner (rung 1: a batch of 99 networks) runs session 1 once; a
snapshot then goes through every arm (an offline phase or none), followed by the same
session 2. The primary outcome is the share of *switchers* (§3 of the protocol).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch

from .harness import rng_for, seed_for
from .learners import (
    G_C,
    W_C,
    GatedBuffer,
    GatedConfig,
    GatedNet,
    NRTBuffer,
    NRTConfig,
    NRTLearner,
    analytic_accuracy,
    nrt_loss,
    train_lowe,
)
from .measures import binned_mean, effective_rank, twonn_dimension
from .phases import SleepConfig, Stage, ablation, fit_to_budget, run_stages
from .rung2 import NRTCurriculum, instruct
from .tasks import (
    EARLY_STEP,
    SSSTConfig,
    digit_index,
    load_fitted_motion_means,
    make_nrt_strings,
    make_ssst_trials,
)

# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RungSpec:
    name: str
    lrs: tuple[float, float]
    l1_unit: float  # native per-step L1 shrink of the gates
    noise_unit: float  # native per-step parameter-noise SD (rung 2: chosen unit, §10)
    batch: int
    base_steps: int  # awake steps whose full cost defines the FLOP budget


R1 = RungSpec("r1", (0.3, 0.6), l1_unit=0.07, noise_unit=0.6 * 0.05, batch=1, base_steps=200)
R2 = RungSpec("r2", (0.05, 0.1), l1_unit=0.1 * 0.01, noise_unit=0.003, batch=8,
              base_steps=500)

L1_MULTS = (0.0, 0.5, 1.0, 2.0)
NOISE_MULTS = (0.0, 0.5, 1.0, 2.0)
SHRINKS = (0.0, 0.001, 0.003, 0.01)
REM_MULTS = (0.0, 1.0, 2.0, 4.0)
NREM_STEPS, REM_STEPS = 8, 2


def awake_grid(spec: RungSpec) -> list[dict]:
    return [dict(kind="awake", name=f"awake_lr{lr}_l1x{a}_nzx{b}", lr=lr, l1_mult=a,
                 noise_mult=b)
            for lr in spec.lrs for a in L1_MULTS for b in NOISE_MULTS]


def sleep_grid(spec: RungSpec) -> list[dict]:
    return [dict(kind="sleep", name=f"sleep_lr{lr}_shr{a}_remx{b}", lr=lr, shrink=a,
                 rem_mult=b)
            for lr in spec.lrs for a in SHRINKS for b in REM_MULTS]


def ablation_arms(best_sleep: dict) -> list[dict]:
    return [dict(best_sleep, kind="ablation", keep=k, name=f"{k}_only")
            for k in ("replay", "noise", "shrink")]


BASELINE = dict(kind="baseline", name="baseline")
CEILING = dict(kind="ceiling", name="ceiling")


def _sleep_config(arm: dict, spec: RungSpec) -> SleepConfig:
    return SleepConfig(n_cycles=1, nrem_steps=NREM_STEPS, rem_steps=REM_STEPS, lr=arm["lr"],
                       batch_size=spec.batch, shrink_kind="l2", shrink_amount=arm["shrink"],
                       shrink_group="all", noise_sigma=arm["rem_mult"] * spec.noise_unit,
                       noise_group="all")


def flop_budget(spec: RungSpec, learner) -> int:
    """FLOPs of ``base_steps`` full awake steps (replay + noise + shrink)."""
    full = (learner.flops("grad", spec.batch) + learner.flops("perturb")
            + learner.flops("shrink"))
    return spec.base_steps * full


def plan_for(arm: dict, spec: RungSpec, learner) -> list[Stage]:
    """The step plan of an offline arm, FLOP-matched to the awake budget (§2)."""
    budget = flop_budget(spec, learner)
    if arm["kind"] == "awake":
        cycle = [Stage(1, replay=True, lr=arm["lr"], batch_size=spec.batch,
                       noise_sigma=arm["noise_mult"] * spec.noise_unit, noise_group="all",
                       shrink_kind="l1", shrink_amount=arm["l1_mult"] * spec.l1_unit,
                       shrink_group="gates")]
        return fit_to_budget(cycle, learner, budget)
    if arm["kind"] == "sleep":
        return fit_to_budget(_sleep_config(arm, spec).stages()[0], learner, budget)
    if arm["kind"] == "ablation":
        sleep_arm = dict(arm, kind="sleep")
        n = sum(st.n_steps for st in plan_for(sleep_arm, spec, learner))
        return ablation(_sleep_config(arm, spec), arm["keep"], n_steps=n)
    raise ValueError(arm["kind"])


def apply_offline(learner, buffer, arm: dict, spec: RungSpec, rng) -> dict:
    """Run an offline arm in place. Returns its compute ledger (zeros for no phase)."""
    if arm["kind"] in ("baseline", "ceiling"):
        return dict(steps=0, flops=0, replay_steps=0, examples_replayed=0, wall_s=0.0,
                    budget=flop_budget(spec, learner))
    led = run_stages(learner, buffer, plan_for(arm, spec, learner), 1, rng)
    return dict(led.as_dict(), budget=flop_budget(spec, learner))


# ---------------------------------------------------------------------------
# Rung 1
# ---------------------------------------------------------------------------

R1_S1_COLOUR = 100  # colour-phase trials in session 1
R1_LAST = 100  # trials over which the final accuracy is measured
R1_BIN = 50


def plain_replay(spec: RungSpec) -> dict:
    """Awake practice at the native learning rate with no added L1 or noise: the
    reference arm for choosing the offline budget (§10, amendment 9)."""
    return dict(kind="awake", name="plain_replay", lr=spec.lrs[1], l1_mult=0.0,
                noise_mult=0.0)


def r1_batch(key: tuple, control: bool, arms: list[dict], keep_series: bool = False,
             base_steps: int | None = None) -> dict:
    """Session 1 once for 99 networks, then every arm followed by session 2."""
    spec = R1 if base_steps is None else replace(R1, base_steps=base_steps)
    task = SSSTConfig().as_control() if control else SSSTConfig()
    net = GatedConfig()
    rng = rng_for(*key, "session1")
    trials = make_ssst_trials(task, load_fitted_motion_means(), rng)
    tp, on = task.n_pretrain, task.colour_onset
    split, end = on + R1_S1_COLOUR, task.n_trials
    n = trials.y.shape[0]
    hist, _ = train_lowe(np.full((n, 4), net.init), trials, net, rng, 0, split)
    acc1 = analytic_accuracy(hist[:, :-1], trials.mu_m[:, :split], trials.mu_c[:, :split],
                             task.sigma_m, task.sigma_c, net.sigma_eta)
    low = trials.coh == 0
    motion_low = _masked_mean(acc1[:, tp:on], low[:, tp:on])
    easy_end_motion = _masked_mean(acc1[:, on - 100 : on], trials.coh[:, on - 100 : on] == 4)
    buf = GatedBuffer.from_trials(trials, on, split)
    snap = hist[:, -1].copy()

    results = {}
    for arm in arms:
        learner = GatedNet(n, net, snap.copy())
        if arm["kind"] == "ceiling":  # told the shortcut: colour gate open and held, as in
            # the authors' instructed block (gates frozen, weights keep learning)
            learner.theta[:, G_C] = 0.5
            learner.theta[:, W_C] = np.abs(learner.theta[:, W_C])
        ledger = apply_offline(learner, buf, arm, spec, rng_for(*key, "phase", arm["name"]))
        net2 = replace(net, freeze_gates=True) if arm["kind"] == "ceiling" else net
        h2, _ = train_lowe(learner.theta, trials, net2, rng_for(*key, "session2"), split, end)
        acc2 = analytic_accuracy(h2[:, :-1], trials.mu_m[:, split:], trials.mu_c[:, split:],
                                 task.sigma_m, task.sigma_c, net.sigma_eta)
        low2 = low[:, split:]
        final = _masked_mean(acc2[:, -R1_LAST:], low2[:, -R1_LAST:])
        bins = binned_mean(acc2, low2, R1_BIN)
        res = dict(delta=final - motion_low, bin_gain=bins - motion_low[:, None],
                   ledger=ledger)
        if keep_series:  # 12 bins over motion + colour phase, for the Löwe criterion
            acc_all = np.concatenate([acc1[:, tp:], acc2], axis=1)
            res["series"] = binned_mean(acc_all, low[:, tp:], R1_BIN)
        results[arm["name"]] = res
    return dict(key=list(key), control=control, motion_low=motion_low,
                easy_end_motion=easy_end_motion, results=results)


def _masked_mean(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m, v, 0.0).sum(-1) / m.sum(-1)


# ---------------------------------------------------------------------------
# Rung 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R2Design:
    """Rung-2 difficulty, fixed on development seeds before tuning the arms (§4)."""

    m1: int = 1000  # mirror strings in session 1
    m2: int = 2000  # mirror strings in session 2
    gate_l1: float = 0.01
    n_pre: int = 2000  # unstructured strings before the mirror
    last: int = 400  # strings over which the final early-and-correct rate is measured
    bin: int = 500  # timing bins; must divide m2
    n_probe: int = 300


def _online(learner: NRTLearner, digits, resps, rng) -> np.ndarray:
    b = learner.cfg.batch
    out = np.empty(len(digits), dtype=bool)
    for i in range(0, len(digits), b):
        _, c, e = learner.online_trial(digits[i : i + b], resps[i : i + b], rng)
        out[i : i + b] = (c & e).numpy()
    return out


def _idx(values) -> torch.Tensor:
    return torch.as_tensor(digit_index(values))


def r2_session1(key: tuple, control: bool, design: R2Design):
    """Instruction, unstructured practice, then ``m1`` mirror strings (unstructured for
    controls). Returns the learner, its buffer and competence."""
    cfg = NRTConfig(gate_l1=design.gate_l1)
    rng = rng_for(*key, "session1")
    learner = NRTLearner(cfg, seed_for(*key, "init"))
    instruct(learner, NRTCurriculum(), rng)
    d0, r0 = make_nrt_strings(design.n_pre, rng, mirror=False)
    _online(learner, _idx(d0), _idx(r0), rng)
    d1, r1 = make_nrt_strings(design.m1, rng, mirror=not control)
    ec1 = _online(learner, _idx(d1), _idx(r1), rng)
    dt, rt = make_nrt_strings(500, rng_for(*key, "heldout"), mirror=False)
    with torch.no_grad():
        _, rl, _ = learner.net(_idx(dt))
        step_acc = float((rl[:, 1:].argmax(-1) == _idx(rt)).float().mean())
    return learner, NRTBuffer.from_values(d1, r1), dict(step_acc=step_acc,
                                                        early_correct_s1=ec1.tolist())


def r2_learner(key: tuple, control: bool, arms: list[dict], design: R2Design,
               probe_arms: tuple[str, ...] = (), base_steps: int | None = None) -> dict:
    """Session 1 once, then every arm followed by the same session 2."""
    spec = R2 if base_steps is None else replace(R2, base_steps=base_steps)
    learner, buf, info = r2_session1(key, control, design)
    snap = copy.deepcopy(learner.net.state_dict())
    rng2 = rng_for(*key, "session2")
    d2, r2 = make_nrt_strings(design.m2, rng2, mirror=not control)
    d2, r2 = _idx(d2), _idx(r2)
    probe = _idx(make_nrt_strings(design.n_probe, rng_for(*key, "probe"), mirror=True)[0])
    if design.m2 % design.bin:
        raise ValueError("bin must divide m2")

    results = {}
    for arm in arms:
        lrn = NRTLearner(learner.cfg, 0)
        lrn.net.load_state_dict(snap)
        if arm["kind"] == "ceiling":  # told the shortcut: taught on the stored strings
            _teach(lrn, buf, rng_for(*key, "ceiling"))
        ledger = apply_offline(lrn, buf, arm, spec, rng_for(*key, "phase", arm["name"]))
        probes = [] if arm["name"] in probe_arms else None
        ec = np.empty(design.m2, dtype=bool)
        b = lrn.cfg.batch
        for i in range(0, design.m2, b):
            if probes is not None and i % design.bin == 0:
                probes.append(_probe(lrn, probe, i))
            _, c, e = lrn.online_trial(d2[i : i + b], r2[i : i + b], rng2)
            ec[i : i + b] = (c & e).numpy()
        if probes is not None:
            probes.append(_probe(lrn, probe, design.m2))
        bins = ec.reshape(-1, design.bin).mean(axis=1)
        results[arm["name"]] = dict(final=float(ec[-design.last :].mean()),
                                    bins=bins.tolist(), ledger=ledger, probes=probes)
    return dict(key=list(key), control=control, results=results, **info)


def _teach(learner: NRTLearner, buf: NRTBuffer, rng, n: int = 200) -> None:
    opt = torch.optim.Adam(learner.net.parameters(), lr=1e-2, foreach=False)
    for _ in range(n):
        loss, _ = nrt_loss(learner.net, *buf.sample(rng, 32), learner.cfg.final_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def _probe(learner: NRTLearner, probe: torch.Tensor, i: int) -> dict:
    h = learner.hidden_states(probe)[:, EARLY_STEP - 1].numpy().astype(float)
    return dict(trial=i, twonn=twonn_dimension(h), erank=effective_rank(h))


def r2_calibration_run(key: tuple, gate_l1: float, n_mirror: int) -> list[bool]:
    """Baseline only: one long mirror phase after session-1 practice (difficulty §4)."""
    design = R2Design(m1=n_mirror, gate_l1=gate_l1)
    _, _, info = r2_session1(key, False, design)
    return info["early_correct_s1"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

R1_MIN_GAIN, R2_MIN_RATE = 0.10, 0.5


def thresholds(control_values: dict[str, np.ndarray], floor: float) -> dict[str, float]:
    """Per-arm switch threshold: max(99th percentile of controls, minimum effect)."""
    return {a: float(max(np.nanquantile(v, 0.99), floor)) for a, v in control_values.items()}


def switchers(values: np.ndarray, threshold: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.asarray(values) > threshold


def r1_collect(batches: list[dict]) -> dict[str, np.ndarray]:
    """Arm -> array (n_batches, 99) of deltas."""
    arms = batches[0]["results"].keys()
    return {a: np.array([b["results"][a]["delta"] for b in batches], float) for a in arms}


def r2_collect(learners: list[dict]) -> dict[str, np.ndarray]:
    arms = learners[0]["results"].keys()
    return {a: np.array([x["results"][a]["final"] for x in learners], float) for a in arms}


def shares(test: dict[str, np.ndarray], ctrl: dict[str, np.ndarray], floor: float) -> dict:
    thr = thresholds({a: v.ravel() for a, v in ctrl.items()}, floor)
    return {a: dict(threshold=thr[a], share=float(np.mean(switchers(v, thr[a]))))
            for a, v in test.items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> dict:
    """CI of mean(a - b) over paired units (rows), 2.5/97.5 percentiles."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(boots, 0.025)),
                hi=float(np.quantile(boots, 0.975)))
