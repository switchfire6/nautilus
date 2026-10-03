"""Rung 2: characterise the NRT learner (S1 engineering check, not a claim).

Runs N mirror learners and N control learners and reports insight (Löwe-style
classification of early-and-correct answering), delay, accuracy, compression series
and lead times. Results: runs/rung2/ (cached) and reports/rung2_characterisation.json.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/characterise_rung2.py [--n 12]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nautilus.harness import RUNS_DIR, cached, run_parallel, save_json, seed_for
from nautilus.learners import NRTConfig
from nautilus.measures import (
    binned_mean,
    classify_insight,
    fit_switch,
    lead_time,
    onset_hinge,
)
from nautilus.rung2 import NRTCurriculum, run_nrt
from nautilus.tasks import REPO_ROOT

OUT = RUNS_DIR / "rung2"
CUR = NRTCurriculum(n_pre=2000, n_main=6000, probe_every=200)
BIN = 400  # strings per bin
N_PRE_BINS = 4  # bins before mirror onset in the analysis window


def run_one(job: dict) -> dict:
    def compute():
        t0 = time.perf_counter()
        r = run_nrt(NRTConfig(), CUR, seed_for("rung2", job["seed"], job["control"]),
                    control=job["control"])
        ec = r["early_correct"].astype(float)
        start = CUR.n_pre - N_PRE_BINS * BIN
        series = binned_mean(ec[start:], np.ones(len(ec) - start, bool), BIN)
        return dict(job=job, series=series, acc=float(r["correct"][CUR.n_pre:].mean()),
                    probes={k: v.tolist() for k, v in r["probes"].items()},
                    wall_s=time.perf_counter() - t0)

    return cached(OUT / f"seed={job['seed']}_control={job['control']}.json", compute)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [dict(seed=s, control=c) for c in (False, True) for s in range(args.n)]
    t0 = time.perf_counter()
    res = run_parallel(run_one, jobs, args.workers)
    test = [r for r in res if not r["job"]["control"]]
    ctrl = [r for r in res if r["job"]["control"]]
    n_bins = len(test[0]["series"])
    bounds = ((0.0, 1.0), (0.0, n_bins), (0.0, 9.0))
    fits_t = [fit_switch(np.array(r["series"]), N_PRE_BINS, (0.5, N_PRE_BINS, 7.0),
                         bounds, "global") for r in test]
    fits_c = [fit_switch(np.array(r["series"]), N_PRE_BINS, (0.5, N_PRE_BINS, 7.0),
                         bounds, "global") for r in ctrl]
    insight = classify_insight([f.score for f in fits_t], [f.score for f in fits_c])
    final_early = np.array([r["series"][-1] for r in test])

    # lead time: compression onset (hinge on the main-phase probe series) minus the
    # behavioural switch, both in strings since mirror onset
    leads: dict[str, list] = {"twonn": [], "corrdim": [], "erank": []}
    for r, f, ins in zip(test, fits_t, insight):
        if not ins:
            continue
        trials = np.array(r["probes"]["trial"])
        main = trials >= CUR.n_pre
        switch = (f.t_switch - N_PRE_BINS - 0.5) * BIN  # bin centre -> strings after onset
        for k in leads:
            y = np.array(r["probes"][k])[main]
            onset = trials[main][onset_hinge(y)] - CUR.n_pre
            leads[k].append(lead_time(onset, switch))

    summary = dict(
        n_learners=args.n,
        n_insight=int(insight.sum()),
        n_final_early_gt_half=int((final_early > 0.5).sum()),
        control_max_score=float(max(f.score for f in fits_c)),
        control_max_early=float(max(max(r["series"]) for r in ctrl)),
        switch_strings_after_onset=[float((f.t_switch - N_PRE_BINS - 0.5) * BIN)
                                    for f, i in zip(fits_t, insight) if i],
        acc_main_phase=dict(test=float(np.mean([r["acc"] for r in test])),
                            control=float(np.mean([r["acc"] for r in ctrl]))),
        final_early_correct=final_early.tolist(),
        lead_time_strings={k: dict(values=v, median=float(np.median(v)) if v else None)
                           for k, v in leads.items()},
        wall_s_per_learner=float(np.mean([r["wall_s"] for r in res])),
        wall_s_total=time.perf_counter() - t0,
        curriculum=CUR.__dict__,
        config=NRTConfig().__dict__,
    )
    digest = save_json(REPO_ROOT / "reports/rung2_characterisation.json", summary)
    print(f"sha256 {digest[:16]}")
    for k, v in summary.items():
        if k not in ("curriculum", "config"):
            print(k, v)


if __name__ == "__main__":
    main()
