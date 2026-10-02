import numpy as np

from nautilus.learners import W_C, GatedConfig
from nautilus.rung1 import replicate_batch, simulate
from nautilus.tasks import SSSTConfig, load_fitted_motion_means

MU = load_fitted_motion_means()[:12]


def test_simulate_shapes():
    out = simulate(SSSTConfig(), GatedConfig(), MU, np.random.default_rng(0))
    t = SSSTConfig().n_trials
    assert out["hist"].shape == (12, t + 1, 4)
    assert out["acc"].shape == (12, t)
    assert out["series"].shape == (12, 12)
    assert np.all((out["acc"] >= 0) & (out["acc"] <= 1))


def test_no_update_noise_no_silent_colour_weight():
    """Without update noise the colour weight cannot grow while its gate is shut."""
    net = GatedConfig(sigma_xi=0.0)
    out = simulate(SSSTConfig(), net, MU, np.random.default_rng(1))
    on = SSSTConfig().colour_onset
    assert np.all(np.abs(out["hist"][:, on, W_C]) < 0.05)


def test_replicate_batch_fields_and_determinism():
    a = replicate_batch(SSSTConfig(), GatedConfig(), MU, np.random.default_rng(2), ("cobyla",))
    b = replicate_batch(SSSTConfig(), GatedConfig(), MU, np.random.default_rng(2), ("cobyla",))
    r = a["cobyla"]
    assert r["n"] == 12 and 0 <= r["n_insight"] <= 12
    assert len(r["scores"]) == 12 and len(r["control_scores"]) == 12
    assert a == b


def test_networks_learn_motion():
    out = simulate(SSSTConfig(), GatedConfig(), MU, np.random.default_rng(3))
    on = SSSTConfig().colour_onset
    easy = out["trials"].coh[:, :on] == 4
    late = np.zeros_like(easy)
    late[:, on - 200 : on] = True
    assert out["acc"][:, :on][easy & late].mean() > 0.75
