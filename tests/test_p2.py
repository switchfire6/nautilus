import numpy as np

from nautilus import p2


def test_auc():
    assert p2.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert p2.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert p2.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_cv_auc_detects_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(bool)
    x_good = y + rng.normal(0, 0.5, 300)
    x_noise = rng.normal(size=300)
    assert p2.cv_auc(x_good, y) > 0.8
    assert abs(p2.cv_auc(x_noise, y) - 0.5) < 0.1


def test_predictors_signs():
    run = dict(probes=[dict(trial=0, twonn=5.0, corrdim=2.0, erank=10.0, p_correct=0.3,
                            move=0.0),
                       dict(trial=p2.WINDOW, twonn=4.0, corrdim=1.5, erank=12.0,
                            p_correct=0.5, move=2.0)])
    pr = p2.predictors(run)
    assert pr["P_twonn"] == 1.0 and pr["P_corr"] == 0.5 and pr["P_rank"] == -2.0
    assert np.isclose(pr["K_silent"], 0.2) and pr["K_move"] == 2.0


def test_probe_times():
    t = p2.probe_times()
    assert t[0] == 0 and p2.WINDOW in t and t[-1] == p2.N_MIRROR
