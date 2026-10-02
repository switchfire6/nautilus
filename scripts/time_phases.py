"""Timings and compute accounting of awake, sleep and ablation phases on both rungs.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/time_phases.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from nautilus.harness import init_worker, save_json
from nautilus.learners import GatedBuffer, GatedConfig, GatedNet, NRTBuffer, NRTLearner
from nautilus.phases import (
    AwakeConfig,
    SleepConfig,
    check_equal_budget,
    run_ablation,
    run_awake,
    run_sleep,
)
from nautilus.rung1 import simulate
from nautilus.tasks import REPO_ROOT, SSSTConfig, load_fitted_motion_means, make_nrt_strings

N_STEPS = 1000


def arms(lr: float, bs: int) -> tuple[AwakeConfig, SleepConfig]:
    awake = AwakeConfig(n_steps=N_STEPS, lr=lr, batch_size=bs, reg_strength=0.05,
                        noise_sigma=0.01)
    sleep = SleepConfig(n_cycles=N_STEPS // 10, nrem_steps=8, rem_steps=2, lr=lr,
                        batch_size=bs, shrink_kind="l2", shrink_amount=0.001,
                        noise_sigma=0.05)
    return awake, sleep


def time_all(make_learner, buf, lr, bs) -> dict:
    awake, sleep = arms(lr, bs)
    rng = np.random.default_rng(0)
    run_awake(make_learner(), buf, awake, rng)  # warm-up
    out = {}
    for name, fn in [
        ("awake", lambda lrn: run_awake(lrn, buf, awake, rng)),
        ("sleep", lambda lrn: run_sleep(lrn, buf, sleep, rng)),
        ("replay_only", lambda lrn: run_ablation(lrn, buf, sleep, "replay", rng)),
        ("noise_only", lambda lrn: run_ablation(lrn, buf, sleep, "noise", rng)),
        ("shrink_only", lambda lrn: run_ablation(lrn, buf, sleep, "shrink", rng)),
    ]:
        out[name] = fn(make_learner())
    res = {k: dict(v.as_dict(), us_per_step=1e6 * v.wall_s / v.steps) for k, v in out.items()}
    res["budget_awake_vs_sleep"] = check_equal_budget(out["awake"], out["sleep"])
    return res


def main() -> None:
    init_worker()
    report = {}

    # rung 1: 99 gated networks after the standard curriculum
    mu = load_fitted_motion_means()
    t0 = time.perf_counter()
    sim = simulate(SSSTConfig(), GatedConfig(), mu, np.random.default_rng(0))
    report["rung1_simulate_99_nets_s"] = time.perf_counter() - t0
    on = SSSTConfig().colour_onset
    buf = GatedBuffer.from_trials(sim["trials"], on, on + 200)
    theta = sim["hist"][:, on + 200]
    report["rung1_phases_99_nets"] = time_all(
        lambda: GatedNet(99, GatedConfig(), theta.copy()), buf, lr=0.6, bs=1)

    # rung 2: one NRT learner, buffer of 1000 stored strings
    d, r = make_nrt_strings(1000, np.random.default_rng(1), mirror=True)
    nbuf = NRTBuffer.from_values(d, r)

    def make_nrt():
        return NRTLearner(seed=0)

    report["rung2_phases_1_learner"] = time_all(make_nrt, nbuf, lr=0.1, bs=8)
    report["torch_threads"] = torch.get_num_threads()
    save_json(REPO_ROOT / "reports/timings.json", report)
    for rung in ("rung1_phases_99_nets", "rung2_phases_1_learner"):
        print(rung)
        for k, v in report[rung].items():
            print(f"  {k:24s}", {kk: (round(vv, 1) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items()})


if __name__ == "__main__":
    main()
