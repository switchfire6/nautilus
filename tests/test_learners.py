import numpy as np
import pytest
import torch

from nautilus.learners import (
    G_C,
    G_M,
    W_C,
    W_M,
    GatedBuffer,
    GatedConfig,
    GatedNet,
    NRTBuffer,
    NRTConfig,
    NRTLearner,
    analytic_accuracy,
    answer_readout,
    data_grad,
    lowe_step,
    nrt_loss,
    soft_threshold,
)
from nautilus.tasks import make_nrt_strings


def test_data_grad_matches_finite_difference():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(3, 4))
    xm, xc, y, eta = (rng.normal(size=(3, 1)) for _ in range(4))

    def loss(th):
        return 0.5 * (th[:, 1:2] * th[:, 0:1] * xm + th[:, 3:4] * th[:, 2:3] * xc + eta - y) ** 2

    g = data_grad(theta, xm, xc, y, eta)[:, 0]
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1e-6
        fd = (loss(theta + e) - loss(theta - e))[:, 0] / 2e-6
        assert np.allclose(g[:, k], fd, atol=1e-6)


def test_lowe_step_code_rule():
    cfg = GatedConfig(clip_gates=False)
    theta = np.array([[1.0, 0.5, 0.3, 0.2]])
    one = np.ones(1)
    xi = np.full((1, 4), 0.1)
    new, grad = lowe_step(theta, one * 0.4, one * 0.2, one, one * 0.0, xi, cfg)
    expect = theta - 0.6 * grad
    expect[:, [G_M, G_C]] -= 0.07  # lam * sign(g), not scaled by alpha
    expect += 0.6 * 0.1  # alpha * xi
    assert np.allclose(new, expect)


def test_lowe_step_paper_rule():
    cfg = GatedConfig.paper()
    theta = np.array([[1.0, 0.5, 0.3, 0.2]])
    one = np.ones(1)
    xi = np.full((1, 4), 0.1)
    new, grad = lowe_step(theta, one * 0.4, one * 0.2, one, one * 0.0, xi, cfg)
    expect = theta - 0.6 * grad
    expect[:, [G_M, G_C]] -= 0.6 * 0.07
    expect += 0.1
    assert np.allclose(new, expect)


def test_lowe_gate_clipping():
    cfg = GatedConfig()
    one = np.ones(1)
    zero_inputs = (one * 0, one * 0, one * 0, one * 0)
    # a small positive gate overshoots below zero under L1 -> clipped to exactly 0
    theta = np.array([[0.0, 0.01, 0.0, 0.01]])
    new, _ = lowe_step(theta, *zero_inputs, np.zeros((1, 4)), cfg)
    assert new[0, G_M] == 0.0 and new[0, G_C] == 0.0
    # a gate at 0 stays at 0 when its noise step is below alpha*lam = 0.042
    theta = np.zeros((1, 4))
    xi = np.array([[0.0, 0.05, 0.0, -0.05]])  # alpha*xi = +-0.03
    new, _ = lowe_step(theta, *zero_inputs, xi, cfg)
    assert new[0, G_M] == 0.0 and new[0, G_C] == 0.0
    # ... and moves by (step - alpha*lam) when the step is larger
    xi = np.array([[0.0, 0.1, 0.0, 0.0]])  # alpha*xi = 0.06
    new, _ = lowe_step(theta, *zero_inputs, xi, cfg)
    assert new[0, G_M] == pytest.approx(0.06 - 0.042)


def test_analytic_accuracy():
    th = np.array([0.0, 0.0, 0.0, 0.0])
    assert analytic_accuracy(th, 0.2, 0.0, 0.1, 0.01, 0.3) == pytest.approx(0.5)
    th = np.array([20.0, 1.0, 0.0, 0.0])
    assert analytic_accuracy(th, 0.4, 0.0, 0.1, 0.01, 0.3) > 0.99
    # matches simulation for +-1 targets
    rng = np.random.default_rng(1)
    th = np.array([3.0, 0.5, 2.0, 0.3])
    mu_m, mu_c, sm, sc, se = 0.1, 0.22, 0.1, 0.01, 0.3
    n = 200_000
    xm = rng.normal(mu_m, sm, n)
    xc = rng.normal(mu_c, sc, n)
    out = th[0] * th[1] * xm + th[2] * th[3] * xc + rng.normal(0, se, n)
    assert np.mean(out > 0) == pytest.approx(
        analytic_accuracy(th, mu_m, mu_c, sm, sc, se), abs=0.005)


def test_gated_net_phase_ops():
    rng = np.random.default_rng(2)
    net = GatedNet(5, GatedConfig(), theta=rng.normal(size=(5, 4)))
    before = net.theta.copy()
    net.perturb(0.1, "gates", rng)
    assert np.allclose(net.theta[:, [W_M, W_C]], before[:, [W_M, W_C]])
    assert not np.allclose(net.theta[:, [G_M, G_C]], before[:, [G_M, G_C]])
    before = net.theta.copy()
    net.shrink("l2", 0.5, "weights")
    assert np.allclose(net.theta[:, [W_M, W_C]], 0.5 * before[:, [W_M, W_C]])
    assert np.allclose(net.theta[:, [G_M, G_C]], before[:, [G_M, G_C]])
    net.shrink("l1", 10.0, "all")
    assert np.all(net.theta == 0)
    assert net.flops("grad", 4) > net.flops("perturb")


def test_gated_grad_step_reduces_loss():
    rng = np.random.default_rng(3)
    n, m = 4, 200
    y = rng.integers(0, 2, size=(n, m)).astype(float)
    buf = GatedBuffer(xm=y * 0.3 + rng.normal(0, 0.05, (n, m)),
                      xc=rng.normal(0, 0.01, (n, m)), y=y)
    net = GatedNet(n, GatedConfig(sigma_eta=0.0), theta=np.full((n, 4), 0.5))

    def loss():
        return np.mean(data_grad(net.theta, buf.xm, buf.xc, buf.y, 0.0)[..., 0] ** 2)

    before = loss()
    for _ in range(200):
        net.grad_step(buf.sample(rng, 8), 0.5, rng)
    assert loss() < before


def test_soft_threshold_numpy_and_torch():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert soft_threshold(x, 1.0).tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]
    assert soft_threshold(torch.tensor(x), 1.0).tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_answer_readout():
    big = 10.0
    logits = torch.zeros(3, 8, 3)
    logits[0, 2, 1] = big  # confident at step 3 -> early
    logits[1, 5, 2] = big  # confident at step 6
    # learner 2 is never confident -> answers at step 8 with argmax (class 0 by tie)
    answer = torch.tensor([1, 0, 0])
    step, correct, early = answer_readout(logits, answer, 0.9)
    assert step.tolist() == [3, 6, 8]
    assert correct.tolist() == [True, False, True]
    assert early.tolist() == [True, False, False]


def test_nrt_learner_ops():
    learner = NRTLearner(NRTConfig(hidden=8), seed=0)
    d, r = make_nrt_strings(16, np.random.default_rng(0), mirror=False)
    buf = NRTBuffer.from_values(d, r)
    h, rl, fl = learner.net(buf.digits)
    assert h.shape == (16, 8, 8) and rl.shape == (16, 8, 3) and fl.shape == (16, 8, 3)
    before, _ = nrt_loss(learner.net, buf.digits, buf.resps, 1.0)
    for _ in range(20):
        learner.grad_step((buf.digits, buf.resps), 0.5, np.random.default_rng(0))
    after, _ = nrt_loss(learner.net, buf.digits, buf.resps, 1.0)
    assert after < before
    w_before = learner.net.resp.weight.detach().clone()
    learner.shrink("l1", 100.0, "gates")
    assert torch.all(learner.net.gate == 0)
    assert torch.equal(learner.net.resp.weight, w_before)
    learner.perturb(0.1, "weights", np.random.default_rng(1))
    assert not torch.equal(learner.net.resp.weight, w_before)
    assert torch.all(learner.net.gate == 0)
    assert learner.flops("grad", 4) > learner.flops("grad", 1) > learner.flops("shrink")


def test_freeze_gates():
    cfg = GatedConfig(freeze_gates=True)
    theta = np.array([[1.0, 0.5, 0.3, 0.2]])
    one = np.ones(1)
    new, _ = lowe_step(theta, one * 0.4, one * 0.2, one, one * 0.0, np.full((1, 4), 0.1), cfg)
    assert new[0, G_M] == 0.5 and new[0, G_C] == 0.2
    assert new[0, W_M] != 1.0
