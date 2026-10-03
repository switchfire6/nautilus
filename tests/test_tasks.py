import numpy as np
import pytest

from nautilus.tasks import (
    COH_PER_100,
    DIGITS,
    PHASE_COLOUR,
    PHASE_MOTION,
    PHASE_PRETRAIN,
    SSSTConfig,
    _balanced_block,
    digit_index,
    load_fitted_motion_means,
    make_nrt_strings,
    make_ssst_trials,
    nrt_responses,
    nrt_rule,
)

MU = np.array([[0.05, 0.1, 0.2, 0.3, 0.45]] * 4)


def test_fitted_means_table():
    mu = load_fitted_motion_means()
    assert mu.shape == (99, 5)
    assert np.all(mu >= 0)


def test_balanced_block_counts():
    coh, lab = _balanced_block(np.random.default_rng(0), 100)
    for k, c in enumerate(COH_PER_100):
        assert np.sum(coh == k) == c
        assert np.sum((coh == k) & (lab == 1)) == c // 2
    assert lab.sum() == 50


def test_ssst_curriculum_and_inputs():
    cfg = SSSTConfig()
    tr = make_ssst_trials(cfg, MU, np.random.default_rng(1))
    assert tr.xm.shape == (4, cfg.n_trials)
    assert np.all(tr.phase[: cfg.n_pretrain] == PHASE_PRETRAIN)
    assert np.all(tr.phase[cfg.n_pretrain : cfg.colour_onset] == PHASE_MOTION)
    assert np.all(tr.phase[cfg.colour_onset :] == PHASE_COLOUR)
    assert set(np.unique(tr.coh[:, : cfg.n_pretrain])) <= {2, 3, 4}
    assert set(np.unique(tr.y)) == {0.0, 1.0}
    on = cfg.colour_onset
    # colour carries the answer only in the colour phase
    pos, neg = tr.y[:, on:] == 1, tr.y[:, on:] == 0
    assert tr.xc[:, on:][pos].mean() == pytest.approx(cfg.mu_c, abs=0.01)
    assert tr.xc[:, on:][neg].mean() == pytest.approx(0.0, abs=0.01)
    assert abs(tr.xc[:, :on].mean()) < 0.01
    # motion mean follows the coherence level
    assert np.allclose(tr.mu_m, np.take_along_axis(MU, tr.coh, axis=1))


def test_ssst_control_never_predictive():
    cfg = SSSTConfig().as_control()
    tr = make_ssst_trials(cfg, MU, np.random.default_rng(2))
    assert np.all(tr.mu_c == 0)
    assert abs(np.corrcoef(tr.xc.ravel(), tr.y.ravel())[0, 1]) < 0.05


def test_ssst_paper_variant():
    cfg = SSSTConfig.paper()
    tr = make_ssst_trials(cfg, MU, np.random.default_rng(3))
    assert cfg.n_pretrain == 600
    assert set(np.unique(tr.y)) == {-1.0, 1.0}
    on = cfg.colour_onset
    # before onset colour is +-mu_c at random; after onset it equals y*mu_c
    assert np.abs(tr.xc[:, :on]).mean() == pytest.approx(cfg.mu_c, abs=0.01)
    assert np.allclose(np.sign(tr.xc[:, on:]), tr.y[:, on:])


def test_nrt_rule_is_modular_addition():
    for a in DIGITS:
        for b in DIGITS:
            r = nrt_rule(a, b)
            ia, ib, ir = digit_index(np.array([a, b, r]))
            assert ir == (-(ia + ib)) % 3
    assert nrt_rule(4, 4) == 4
    assert nrt_rule(1, 9) == 4


@pytest.mark.parametrize("mirror", [True, False])
def test_nrt_strings_consistent(mirror):
    d, r = make_nrt_strings(300, np.random.default_rng(4), mirror=mirror)
    assert d.shape == (300, 8) and r.shape == (300, 7)
    assert set(np.unique(d)) <= set(DIGITS)
    for di, ri in zip(d, r):
        assert nrt_responses(list(di)) == list(ri)
    if mirror:
        assert np.all(r[:, 1] == r[:, 6])  # 2nd response = final answer
        assert np.all(r[:, 2] == r[:, 5]) and np.all(r[:, 3] == r[:, 4])
    else:
        assert np.mean(r[:, 1] == r[:, 6]) < 0.5


def test_digit_index():
    assert digit_index(np.array([1, 4, 9])).tolist() == [0, 1, 2]
