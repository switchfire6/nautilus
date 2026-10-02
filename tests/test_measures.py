import numpy as np
import pytest

from nautilus.measures import (
    binned_mean,
    change_point_mean,
    classify_insight,
    correlation_dimension,
    effective_rank,
    fit_switch,
    gate_sparsity,
    hoyer_sparsity,
    lead_time,
    onset_hinge,
    sigmoid,
    switch_delay_trials,
    twonn_dimension,
)


def test_binned_mean():
    v = np.arange(8, dtype=float)
    mask = np.array([1, 0, 1, 1, 0, 0, 1, 1], bool)
    out = binned_mean(v, mask, 4)
    assert out.tolist() == [(0 + 2 + 3) / 3, (6 + 7) / 2]
    assert np.isnan(binned_mean(v, np.zeros(8, bool), 4)).all()


@pytest.mark.parametrize("method", ["cobyla", "global"])
def test_fit_switch_recovers_sigmoid(method):
    t = np.arange(1, 13)
    y = sigmoid(t, 0.6, 0.95, 2.0, 7.0)
    y[:4] = 0.6
    f = fit_switch(y, method=method)
    assert f.fmin == pytest.approx(0.6)
    assert f.t_switch == pytest.approx(7.0, abs=0.3)
    assert f.fmax == pytest.approx(0.95, abs=0.03)
    if method == "global":
        assert f.slope == pytest.approx(2.0, rel=0.1)
        assert f.sse < 1e-4


def test_fit_switch_flat_series_has_low_score():
    f = fit_switch(np.full(12, 0.6), method="global")
    assert abs(f.steepness) < 0.02


def test_classify_and_delay():
    assert classify_insight([0.1, 0.5, 0.3], [0.2, 0.3]).tolist() == [False, True, False]
    nan = float("nan")
    assert classify_insight([0.5, nan], [0.2, nan]).tolist() == [True, False]
    assert switch_delay_trials(7.5) == pytest.approx(175.0)


def test_twonn_dimension():
    rng = np.random.default_rng(0)
    plane = rng.uniform(size=(800, 2)) @ rng.normal(size=(2, 10))
    assert twonn_dimension(plane) == pytest.approx(2.0, abs=0.3)
    cube = rng.uniform(size=(1500, 5))
    assert twonn_dimension(cube) == pytest.approx(5.0, abs=0.8)
    assert np.isnan(twonn_dimension(np.zeros((10, 3))))  # all duplicates


def test_correlation_dimension():
    rng = np.random.default_rng(1)
    square = rng.uniform(size=(1500, 2))
    assert correlation_dimension(square) == pytest.approx(2.0, abs=0.3)
    line = np.outer(rng.uniform(size=1000), rng.normal(size=6))
    assert correlation_dimension(line) == pytest.approx(1.0, abs=0.15)


def test_effective_rank():
    rng = np.random.default_rng(2)
    rank1 = np.outer(rng.normal(size=200), rng.normal(size=8))
    assert effective_rank(rank1) == pytest.approx(1.0, abs=1e-6)
    iso = rng.normal(size=(5000, 6))
    assert effective_rank(iso) == pytest.approx(6.0, abs=0.2)


def test_sparsity():
    g = np.array([0.0, 1e-4, 0.5, -0.2])
    assert gate_sparsity(g, 1e-3) == 0.5
    assert hoyer_sparsity(np.array([0, 0, 3.0])) == pytest.approx(1.0)
    assert hoyer_sparsity(np.ones(5)) == pytest.approx(0.0)


def test_change_points_and_lead_time():
    step = np.r_[np.zeros(10), np.ones(10)]
    assert change_point_mean(step) == 10
    hinge = np.r_[np.full(8, 3.0), 3.0 - 0.5 * np.arange(1, 13)]
    assert onset_hinge(hinge) == 7  # last constant index; decline starts after it
    assert lead_time(5.0, 8.0) == -3.0
