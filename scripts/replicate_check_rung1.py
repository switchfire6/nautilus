"""Replicate pass for rung 1: recompute a subset of cached batches from scratch and
compare *results* (insight labels and scores), not metadata.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/replicate_check_rung1.py
"""

from __future__ import annotations

import json

import numpy as np
from replicate_rung1 import OUT, configs, job_name

from nautilus.harness import rng_for, run_parallel
from nautilus.rung1 import replicate_batch
from nautilus.tasks import load_fitted_motion_means

SUBSET = [dict(group="main", variant="code", batch=b) for b in (0, 7)] + [
    dict(group="lam", variant="code", lam=0.03, batch=2)]


def redo(j: dict) -> dict:
    task, net = configs(j)
    return replicate_batch(task, net, load_fitted_motion_means(),
                           rng_for("rung1", job_name(j)))


def main() -> None:
    ok = True
    for j, new in zip(SUBSET, run_parallel(redo, SUBSET, 3)):
        old = json.loads((OUT / f"{job_name(j)}.json").read_text())["result"]
        for m in ("cobyla", "global"):
            same = old[m]["insight_mask"] == new[m]["insight_mask"]
            diff = float(np.max(np.abs(np.subtract(old[m]["scores"], new[m]["scores"]))))
            ok &= same and diff < 1e-9
            print(f"{job_name(j)} {m}: n_insight {old[m]['n_insight']} -> "
                  f"{new[m]['n_insight']}, labels identical {same}, max|d score| {diff:.1e}")
    print("REPLICATE PASS" if ok else "REPLICATE MISMATCH")


if __name__ == "__main__":
    main()
