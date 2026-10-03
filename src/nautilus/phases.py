"""Offline phases after a learning session: awake practice, sleep, and sleep ablations.

All phases are built from one engine, ``run_stages``. A *stage* is a block of update
steps; each step can contain any of three ingredients:

- **replay**: an SGD step on a batch of *stored* experiences (no new data);
- **noise**: a Gaussian perturbation of a parameter group;
- **shrink**: scaling a parameter group down (L1 soft-threshold or L2 multiplicative).

The arms differ only in how the ingredients are arranged:

- **awake** is one stage repeated: every step replays stored experiences, with the
  learner's own continuous regulariser and noise (its strengths are tunable);
- **sleep** is a cycle of stages, by default an NREM-like stage (replay + scaling-down)
  followed by a REM-like stage (noise, no replay);
- **ablations** keep a single sleep ingredient.

Every step is counted as one update step, and its FLOPs are added to a ``ComputeLedger``.
``check_equal_budget`` verifies (does not assume) that two arms got the same compute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .learners import PhaseLearner


@dataclass(frozen=True)
class Stage:
    n_steps: int
    replay: bool = False
    lr: float = 0.0
    batch_size: int = 1
    noise_sigma: float = 0.0
    noise_group: str = "all"
    shrink_kind: str = "l1"  # "l1" | "l2"
    shrink_amount: float = 0.0
    shrink_group: str = "gates"

    def step_flops(self, learner: PhaseLearner) -> int:
        f = 0
        if self.replay:
            f += learner.flops("grad", self.batch_size)
        if self.noise_sigma > 0:
            f += learner.flops("perturb")
        if self.shrink_amount > 0:
            f += learner.flops("shrink")
        return f


@dataclass
class ComputeLedger:
    steps: int = 0
    flops: int = 0
    replay_steps: int = 0
    examples_replayed: int = 0
    wall_s: float = 0.0
    by_stage: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(
            steps=self.steps,
            flops=self.flops,
            replay_steps=self.replay_steps,
            examples_replayed=self.examples_replayed,
            wall_s=self.wall_s,
        )


def run_stages(
    learner: PhaseLearner,
    buffer,
    stages: list[Stage],
    n_cycles: int,
    rng: np.random.Generator,
    ledger: ComputeLedger | None = None,
) -> ComputeLedger:
    """Run ``stages`` in order, ``n_cycles`` times. ``buffer`` is only read (``sample``)."""
    ledger = ComputeLedger() if ledger is None else ledger
    t0 = time.perf_counter()
    for _ in range(n_cycles):
        for st in stages:
            for _ in range(st.n_steps):
                if st.replay:
                    learner.grad_step(buffer.sample(rng, st.batch_size), st.lr, rng)
                if st.noise_sigma > 0:
                    learner.perturb(st.noise_sigma, st.noise_group, rng)
                if st.shrink_amount > 0:
                    learner.shrink(st.shrink_kind, st.shrink_amount, st.shrink_group)
            ledger.steps += st.n_steps
            ledger.flops += st.n_steps * st.step_flops(learner)
            if st.replay:
                ledger.replay_steps += st.n_steps
                ledger.examples_replayed += st.n_steps * st.batch_size
            ledger.by_stage.append((st, st.n_steps))
    ledger.wall_s += time.perf_counter() - t0
    return ledger


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AwakeConfig:
    """Extra practice on stored experiences with continuous regularisation and noise."""

    n_steps: int
    lr: float
    batch_size: int = 1
    reg_kind: str = "l1"
    reg_strength: float = 0.0  # per-step shrink amount is lr * reg_strength
    reg_group: str = "gates"
    noise_sigma: float = 0.0  # per-step parameter noise SD
    noise_group: str = "all"

    def stages(self) -> tuple[list[Stage], int]:
        st = Stage(
            n_steps=1,
            replay=True,
            lr=self.lr,
            batch_size=self.batch_size,
            noise_sigma=self.noise_sigma,
            noise_group=self.noise_group,
            shrink_kind=self.reg_kind,
            shrink_amount=self.lr * self.reg_strength,
            shrink_group=self.reg_group,
        )
        return [st], self.n_steps


@dataclass(frozen=True)
class SleepConfig:
    """Offline cycles of an NREM-like stage (replay + scaling-down) and a REM-like stage
    (noise). Setting an ingredient's strength to 0 removes it."""

    n_cycles: int
    nrem_steps: int
    rem_steps: int
    lr: float
    batch_size: int = 1
    replay: bool = True
    shrink_kind: str = "l2"
    shrink_amount: float = 0.0  # per NREM step
    shrink_group: str = "all"
    noise_sigma: float = 0.0  # per REM step
    noise_group: str = "all"

    @property
    def n_steps(self) -> int:
        return self.n_cycles * (self.nrem_steps + self.rem_steps)

    def stages(self) -> tuple[list[Stage], int]:
        nrem = Stage(
            n_steps=self.nrem_steps,
            replay=self.replay,
            lr=self.lr,
            batch_size=self.batch_size,
            shrink_kind=self.shrink_kind,
            shrink_amount=self.shrink_amount,
            shrink_group=self.shrink_group,
        )
        rem = Stage(
            n_steps=self.rem_steps,
            noise_sigma=self.noise_sigma,
            noise_group=self.noise_group,
        )
        return [s for s in (nrem, rem) if s.n_steps > 0], self.n_cycles


def ablation(sleep: SleepConfig, keep: str, n_steps: int | None = None) -> list[Stage]:
    """Single-ingredient version of ``sleep`` with ``n_steps`` steps (default: the same
    number of steps as ``sleep``).

    ``keep`` is ``replay``, ``noise`` or ``shrink``. Each step applies only that
    ingredient, at the strength it has in the full sleep phase.
    """
    n = sleep.n_steps if n_steps is None else n_steps
    if keep == "replay":
        return [Stage(n, replay=True, lr=sleep.lr, batch_size=sleep.batch_size)]
    if keep == "noise":
        frac = sleep.rem_steps / (sleep.nrem_steps + sleep.rem_steps)
        # same total noise variance as in full sleep, spread over all steps
        sigma = sleep.noise_sigma * np.sqrt(frac)
        return [Stage(n, noise_sigma=sigma, noise_group=sleep.noise_group)]
    if keep == "shrink":
        frac = sleep.nrem_steps / (sleep.nrem_steps + sleep.rem_steps)
        if sleep.shrink_kind == "l2":  # same total multiplicative shrink
            amount = 1.0 - (1.0 - sleep.shrink_amount) ** frac
        else:  # same total soft-threshold
            amount = sleep.shrink_amount * frac
        return [
            Stage(n, shrink_kind=sleep.shrink_kind, shrink_amount=amount,
                  shrink_group=sleep.shrink_group)
        ]
    raise ValueError(keep)


def fit_to_budget(cycle: list[Stage], learner: PhaseLearner, flops_budget: float) -> list[Stage]:
    """Repeat ``cycle`` and truncate its last repetition so that the counted FLOPs are as
    close as possible to ``flops_budget`` (to within half of one step's FLOPs).

    Steps that cost no FLOPs (an ingredient switched off) are kept but do nothing.
    """
    per_cycle = sum(st.n_steps * st.step_flops(learner) for st in cycle)
    if per_cycle <= 0:
        raise ValueError("a cycle must cost FLOPs")
    full = int(flops_budget // per_cycle)
    out = list(cycle) * full
    left = flops_budget - full * per_cycle
    for st in cycle:
        f = st.step_flops(learner)
        if f == 0:
            continue
        k = min(st.n_steps, int(np.floor(left / f + 0.5)))
        if k <= 0:
            break
        out.append(replace(st, n_steps=k))
        left -= k * f
        if k < st.n_steps:
            break
    return out


def run_awake(learner, buffer, cfg: AwakeConfig, rng) -> ComputeLedger:
    stages, cycles = cfg.stages()
    return run_stages(learner, buffer, stages, cycles, rng)


def run_sleep(learner, buffer, cfg: SleepConfig, rng) -> ComputeLedger:
    stages, cycles = cfg.stages()
    return run_stages(learner, buffer, stages, cycles, rng)


def run_ablation(learner, buffer, cfg: SleepConfig, keep: str, rng) -> ComputeLedger:
    return run_stages(learner, buffer, ablation(cfg, keep), 1, rng)


def check_equal_budget(a: ComputeLedger, b: ComputeLedger, flops_rtol: float = 0.0) -> dict:
    """Compare two arms' compute. ``equal`` requires equal update steps and FLOPs within
    ``flops_rtol`` (relative)."""
    ratio = a.flops / b.flops if b.flops else float("inf")
    steps_equal = a.steps == b.steps
    flops_equal = abs(a.flops - b.flops) <= flops_rtol * max(a.flops, b.flops)
    return dict(
        steps_a=a.steps,
        steps_b=b.steps,
        flops_a=a.flops,
        flops_b=b.flops,
        flops_ratio=ratio,
        steps_equal=steps_equal,
        flops_equal=flops_equal,
        equal=steps_equal and flops_equal,
    )
