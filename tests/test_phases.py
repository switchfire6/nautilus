import numpy as np
import pytest

from nautilus.learners import GatedBuffer, GatedConfig, GatedNet
from nautilus.phases import (
    AwakeConfig,
    SleepConfig,
    ablation,
    check_equal_budget,
    run_ablation,
    run_awake,
    run_sleep,
)


class CountingBuffer(GatedBuffer):
    """Records every sample drawn, to check phases only ever read stored experiences."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def sample(self, rng, batch_size):
        self.calls += 1
        return super().sample(rng, batch_size)


def make(n=6, m=100, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, m)).astype(float)
    buf = CountingBuffer(xm=y * 0.3 + rng.normal(0, 0.1, (n, m)),
                         xc=y * 0.22 + rng.normal(0, 0.01, (n, m)), y=y)
    net = GatedNet(n, GatedConfig(), theta=rng.normal(0.5, 0.2, size=(n, 4)))
    return net, buf


SLEEP = SleepConfig(n_cycles=5, nrem_steps=8, rem_steps=2, lr=0.3, batch_size=4,
                    shrink_kind="l2", shrink_amount=0.01, noise_sigma=0.05)
AWAKE = AwakeConfig(n_steps=50, lr=0.3, batch_size=4, reg_strength=0.05, noise_sigma=0.01)


def test_awake_and_sleep_step_counts_and_data_use():
    net, buf = make()
    led_a = run_awake(net.copy(), buf, AWAKE, np.random.default_rng(1))
    assert led_a.steps == 50 and led_a.replay_steps == 50 and buf.calls == 50
    buf.calls = 0
    led_s = run_sleep(net.copy(), buf, SLEEP, np.random.default_rng(1))
    assert led_s.steps == SLEEP.n_steps == 50
    assert led_s.replay_steps == 40 == buf.calls  # replay only in NREM
    assert led_s.examples_replayed == 160
    budget = check_equal_budget(led_a, led_s)
    assert budget["steps_equal"]
    # awake does replay + shrink + noise every step; sleep splits them, so FLOPs differ
    assert not budget["flops_equal"] and budget["flops_ratio"] > 1


def test_flops_accounting_is_exact():
    net, buf = make()
    led = run_awake(net, buf, AWAKE, np.random.default_rng(0))
    per = net.flops("grad", 4) + net.flops("perturb") + net.flops("shrink")
    assert led.flops == 50 * per
    budget = check_equal_budget(led, led)
    assert budget["equal"] and budget["flops_ratio"] == 1


def test_sleep_without_replay_reads_no_data():
    net, buf = make()
    cfg = SleepConfig(n_cycles=3, nrem_steps=5, rem_steps=5, lr=0.3, replay=False,
                      shrink_amount=0.1, noise_sigma=0.1)
    led = run_sleep(net, buf, cfg, np.random.default_rng(0))
    assert buf.calls == 0 and led.replay_steps == 0 and led.steps == 30


@pytest.mark.parametrize("keep", ["replay", "noise", "shrink"])
def test_ablations_keep_one_ingredient(keep):
    stages = ablation(SLEEP, keep)
    assert sum(s.n_steps for s in stages) == SLEEP.n_steps
    for s in stages:
        assert s.replay == (keep == "replay")
        assert (s.noise_sigma > 0) == (keep == "noise")
        assert (s.shrink_amount > 0) == (keep == "shrink")
    net, buf = make()
    led = run_ablation(net, buf, SLEEP, keep, np.random.default_rng(0))
    assert led.steps == SLEEP.n_steps
    assert (buf.calls > 0) == (keep == "replay")


def test_shrink_ablation_matches_total_shrink():
    net, buf = make()
    full = SleepConfig(n_cycles=5, nrem_steps=8, rem_steps=2, lr=0.0, replay=False,
                       shrink_kind="l2", shrink_amount=0.01, shrink_group="all")
    a, b = net.copy(), net.copy()
    run_sleep(a, buf, full, np.random.default_rng(0))
    run_ablation(b, buf, full, "shrink", np.random.default_rng(0))
    assert np.allclose(a.theta, b.theta)
    assert np.allclose(a.theta, net.theta * 0.99**40)


def test_noise_ablation_matches_total_variance():
    s = ablation(SLEEP, "noise")[0]
    total_full = SLEEP.n_cycles * SLEEP.rem_steps * SLEEP.noise_sigma**2
    assert s.n_steps * s.noise_sigma**2 == pytest.approx(total_full)


def test_phases_are_deterministic():
    net, buf = make()
    a, b = net.copy(), net.copy()
    run_sleep(a, buf, SLEEP, np.random.default_rng(7))
    run_sleep(b, buf, SLEEP, np.random.default_rng(7))
    assert np.array_equal(a.theta, b.theta)
