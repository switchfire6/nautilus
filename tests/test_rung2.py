import numpy as np

from nautilus.learners import NRTConfig
from nautilus.rung2 import NRTCurriculum, run_nrt

CUR = NRTCurriculum(n_instruct=40, n_pre=48, n_main=48, probe_every=32, n_probe=40)


def test_run_nrt_shapes_and_consistency():
    out = run_nrt(NRTConfig(hidden=8, batch=8), CUR, seed=0)
    n = CUR.n_pre + CUR.n_main
    assert out["answer_step"].shape == (n,)
    assert np.all((out["answer_step"] >= 1) & (out["answer_step"] <= 8))
    assert np.array_equal(out["early"], out["answer_step"] <= 3)
    assert np.array_equal(out["early_correct"], out["early"] & out["correct"])
    assert len(out["probes"]["trial"]) == len(out["probes"]["twonn"]) == 4


def test_run_nrt_deterministic():
    a = run_nrt(NRTConfig(hidden=8, batch=8), CUR, seed=3, control=True)
    b = run_nrt(NRTConfig(hidden=8, batch=8), CUR, seed=3, control=True)
    assert np.array_equal(a["answer_step"], b["answer_step"])
    assert np.allclose(a["gates"], b["gates"])
