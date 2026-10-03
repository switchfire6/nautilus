import numpy as np

from nautilus import p1
from nautilus.learners import GatedConfig, GatedNet


def test_grids_have_equal_tuning_budgets():
    for spec in (p1.R1, p1.R2):
        a, s = p1.awake_grid(spec), p1.sleep_grid(spec)
        assert len(a) == len(s) == 32
        assert len({x["name"] for x in a + s}) == 64


def test_awake_and_sleep_plans_are_flop_matched():
    net = GatedNet(3, GatedConfig())
    budget = p1.flop_budget(p1.R1, net)
    for arm in p1.awake_grid(p1.R1) + p1.sleep_grid(p1.R1):
        plan = p1.plan_for(arm, p1.R1, net)
        flops = sum(st.n_steps * st.step_flops(net) for st in plan)
        assert abs(flops - budget) / budget <= 0.02, arm["name"]


def test_ablation_plans_use_sleep_step_count():
    net = GatedNet(3, GatedConfig())
    sleep = p1.sleep_grid(p1.R1)[-1]
    n = sum(st.n_steps for st in p1.plan_for(sleep, p1.R1, net))
    for arm in p1.ablation_arms(sleep):
        assert sum(st.n_steps for st in p1.plan_for(arm, p1.R1, net)) == n


def test_r1_batch_paired_and_ceiling():
    arms = [p1.BASELINE, p1.CEILING, p1.awake_grid(p1.R1)[5], p1.sleep_grid(p1.R1)[9]]
    out = p1.r1_batch(("test", 0), False, arms, keep_series=True)
    base = out["results"]["baseline"]
    assert base["delta"].shape == (99,) and base["series"].shape == (99, 12)
    assert base["ledger"]["flops"] == 0
    assert out["results"]["ceiling"]["delta"].mean() > base["delta"].mean()
    led = out["results"][arms[2]["name"]]["ledger"]
    assert abs(led["flops"] - led["budget"]) / led["budget"] <= 0.02
    again = p1.r1_batch(("test", 0), False, arms[:1])
    assert np.array_equal(again["results"]["baseline"]["delta"], base["delta"])


def test_thresholds_and_shares():
    ctrl = {"a": np.linspace(0, 0.2, 1000)}
    thr = p1.thresholds(ctrl, 0.1)
    assert thr["a"] > 0.19
    assert p1.thresholds({"a": np.zeros(10)}, 0.5)["a"] == 0.5
    sh = p1.shares({"a": np.array([0.0, 0.3, 0.5])}, ctrl, 0.1)
    assert sh["a"]["share"] == 2 / 3


def test_paired_bootstrap():
    a = np.ones(20)
    b = np.zeros(20)
    ci = p1.paired_bootstrap(a, b, 1000)
    assert ci["lo"] == ci["hi"] == ci["mean"] == 1.0


def test_r2_learner_small():
    design = p1.R2Design(m1=16, m2=16, n_pre=16, last=8, bin=8, n_probe=20)
    arms = [p1.BASELINE, p1.awake_grid(p1.R2)[0]]
    p1_r2 = p1.R2  # keep budget small for the test
    small = p1.RungSpec("r2", p1_r2.lrs, p1_r2.l1_unit, p1_r2.noise_unit, 8, 2)
    p1.R2, saved = small, p1.R2
    try:
        out = p1.r2_learner(("t", 1), False, arms, design, probe_arms=("baseline",))
    finally:
        p1.R2 = saved
    assert set(out["results"]) == {"baseline", arms[1]["name"]}
    assert len(out["results"]["baseline"]["bins"]) == 2
    assert len(out["results"]["baseline"]["probes"]) == 3
    assert out["results"][arms[1]["name"]]["ledger"]["steps"] > 0
