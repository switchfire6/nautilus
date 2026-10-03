"""Rung 1: replicate Löwe et al. (2024), arXiv 2302.11351.

Runs batches of 99 test + 99 control gated networks and classifies insight. Each batch
result is cached under runs/rung1/ (resumable); the summary goes to
reports/rung1_replication.json.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/replicate_rung1.py [--workers 4]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace

import numpy as np

from nautilus.harness import (
    RUNS_DIR,
    cached,
    file_sha256,
    rng_for,
    run_parallel,
    save_json,
)
from nautilus.learners import GatedConfig
from nautilus.rung1 import replicate_batch
from nautilus.tasks import REPO_ROOT, SSSTConfig, load_fitted_motion_means

OUT = RUNS_DIR / "rung1"
N_MAIN, N_VARIANT, N_SWEEP = 10, 3, 5
XI_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5)
LAM_LEVELS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def jobs() -> list[dict]:
    js = [dict(group="main", variant="code", batch=b) for b in range(N_MAIN)]
    js += [dict(group="paper_text", variant="paper", batch=b) for b in range(N_VARIANT)]
    js += [dict(group="pretrain600", variant="code", n_pretrain=600, batch=b)
           for b in range(N_VARIANT)]
    js += [dict(group="xi", variant="code", sigma_xi=x, batch=b)
           for x in XI_LEVELS for b in range(N_SWEEP)]
    js += [dict(group="lam", variant="code", lam=lam, batch=b)
           for lam in LAM_LEVELS for b in range(N_SWEEP)]
    return js


def job_name(j: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in j.items())


def configs(j: dict) -> tuple[SSSTConfig, GatedConfig]:
    task = SSSTConfig.paper() if j["variant"] == "paper" else SSSTConfig()
    net = GatedConfig.paper() if j["variant"] == "paper" else GatedConfig()
    if "n_pretrain" in j:
        task = replace(task, n_pretrain=j["n_pretrain"])
    if "sigma_xi" in j:
        net = replace(net, sigma_xi=j["sigma_xi"])
    if "lam" in j:
        net = replace(net, lam=j["lam"])
    return task, net


def run_job(j: dict) -> dict:
    def compute():
        task, net = configs(j)
        t0 = time.perf_counter()
        res = replicate_batch(task, net, load_fitted_motion_means(),
                              rng_for("rung1", job_name(j)))
        return dict(job=j, task=asdict(task), net=asdict(net), result=res,
                    wall_s=time.perf_counter() - t0)

    return cached(OUT / f"{job_name(j)}.json", compute)


def summarise(results: list[dict]) -> dict:
    out: dict = {}
    for method in ("cobyla", "global"):
        groups: dict = {}
        for r in results:
            j = r["job"]
            key = j["group"] + (f"_{j['sigma_xi']}" if "sigma_xi" in j else "") + (
                f"_{j['lam']}" if "lam" in j else "")
            groups.setdefault(key, []).append(r["result"][method])
        summ = {}
        for key, rs in groups.items():
            n_ins = np.array([x["n_insight"] for x in rs])
            delays = np.concatenate([
                (np.array(x["t_switch"])[np.array(x["insight_mask"], bool)] - 4) * 50
                for x in rs])
            ctrl_max = np.array([x["control_max_score"] for x in rs])

            def pooled(field, grp, rs=rs):
                vals = [x[field][grp] for x in rs if x[field][grp]["n"] > 0]
                n = sum(v["n"] for v in vals)
                return (sum(v["mean"] * v["n"] for v in vals) / n) if n else float("nan")

            summ[key] = dict(
                n_batches=len(rs),
                n_insight_per_batch=n_ins.tolist(),
                n_insight_mean=float(n_ins.mean()),
                n_insight_sd=float(n_ins.std(ddof=1)) if len(rs) > 1 else 0.0,
                delay_trials_mean=float(delays.mean()) if delays.size else float("nan"),
                delay_trials_sd=float(delays.std(ddof=1)) if delays.size > 1 else float("nan"),
                control_max_score=ctrl_max.tolist(),
                silent=dict(
                    {f"{f}_{g}": pooled(f, g)
                     for f in ("abs_wc_onset", "abs_wm_onset", "gc_grad_first5",
                               "abs_gc_onset", "abs_gc_end", "abs_gm_end",
                               "acc_motion_phase_low", "acc_colour_phase_low",
                               "slope_fitted")
                     for g in ("insight", "no_insight")}),
                acc_last_block=float(np.mean([x["acc_last_block"]["mean"] for x in rs])),
            )
        out[method] = summ
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.perf_counter()
    results = run_parallel(run_job, jobs(), args.workers)
    summary = dict(
        summary=summarise(results),
        wall_s_total=time.perf_counter() - t0,
        cpu_s_per_batch=float(np.mean([r["wall_s"] for r in results])),
        n_batches=len(results),
        fitted_means_sha256=file_sha256(REPO_ROOT / "configs/lowe2024_fitted_motion_means.csv"),
    )
    digest = save_json(REPO_ROOT / "reports/rung1_replication.json", summary)
    print(f"done in {summary['wall_s_total']:.0f}s; summary sha256 {digest[:16]}")
    for method, summ in summary["summary"].items():
        print(f"\n== fit method: {method}")
        for key, s in summ.items():
            print(f"{key:22s} insight {s['n_insight_mean']:5.1f} ± {s['n_insight_sd']:4.1f}"
                  f"  delay {s['delay_trials_mean']:6.1f} ± {s['delay_trials_sd']:6.1f}")


if __name__ == "__main__":
    main()
