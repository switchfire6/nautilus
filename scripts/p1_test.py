"""P1 test phase (protocol §5). Fresh test seeds; refuses to run unless the lock matches.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/p1_test.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from nautilus import p1
from nautilus.harness import RUNS_DIR, cached, file_sha256, run_parallel, save_json
from nautilus.tasks import REPO_ROOT

TEST = RUNS_DIR / "p1" / "test"
LOCK = REPO_ROOT / "docs/protocols/P1_lock.json"
N_R1, N_R2 = 20, 96


def check_lock() -> dict:
    lock = json.loads(LOCK.read_text())
    bad = [f for f, h in lock["sha256"].items() if file_sha256(REPO_ROOT / f) != h]
    if bad:
        raise SystemExit(f"lock mismatch, refusing to run: {bad}")
    return lock


def arms_for(sel: dict) -> list[dict]:
    return ([p1.BASELINE, sel["awake"], sel["sleep"]] + p1.ablation_arms(sel["sleep"])
            + [p1.CEILING])


def r1_job(j: dict) -> dict:
    return p1.r1_batch(("P1", "test", "r1", j["control"], j["i"]), j["control"], j["arms"],
                       keep_series=True, base_steps=j["steps"])


def r2_job(j: dict) -> dict:
    probe = () if j["control"] else ("baseline", j["arms"][1]["name"], j["arms"][2]["name"])
    return p1.r2_learner(("P1", "test", "r2", j["control"], j["i"]), j["control"], j["arms"],
                         p1.R2Design(**j["design"]), probe_arms=probe, base_steps=j["steps"])


def run(j: dict) -> dict:
    tag = "ctrl" if j["control"] else "test"
    fn = r1_job if j["rung"] == "r1" else r2_job
    return cached(TEST / f"{j['rung']}_{tag}_{j['i']}.json", lambda: fn(j))


def fresh(j: dict) -> dict:
    """Recompute without the cache, for the replicate pass."""
    fn = r1_job if j["rung"] == "r1" else r2_job
    return json.loads(json.dumps(fn(j), default=lambda o: np.asarray(o).tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    check_lock()
    sel = json.loads((REPO_ROOT / "configs/p1_selected.json").read_text())
    design = json.loads((REPO_ROOT / "configs/p1_design.json").read_text())
    steps = design.pop("base_steps")
    t0 = time.perf_counter()
    jobs = [dict(rung="r2", control=c, i=i, arms=arms_for(sel["r2"]), design=design,
                 steps=steps["r2"])
            for i in range(N_R2) for c in (False, True)]
    jobs += [dict(rung="r1", control=c, i=i, arms=arms_for(sel["r1"]), steps=steps["r1"])
             for i in range(N_R1) for c in (False, True)]
    res = run_parallel(run, jobs, args.workers)
    print(f"test runs done in {time.perf_counter() - t0:.0f}s", flush=True)

    # V5 replicate pass: recompute a subset from scratch and compare raw results
    subset = [k for k, j in enumerate(jobs)
              if (j["rung"] == "r1" and j["i"] in (3, 11) and not j["control"])
              or (j["rung"] == "r2" and j["i"] in (5, 50) and not j["control"])]
    again = run_parallel(fresh, [jobs[k] for k in subset], args.workers)
    checks = []
    for k, new in zip(subset, again):
        old = res[k]
        for a in old["results"]:
            key = "delta" if jobs[k]["rung"] == "r1" else "final"
            same = np.array_equal(np.asarray(old["results"][a][key]),
                                  np.asarray(new["results"][a][key]))
            checks.append(dict(rung=jobs[k]["rung"], i=jobs[k]["i"], arm=a, identical=same))
    rep = dict(checks=checks, passed=all(c["identical"] for c in checks))
    save_json(TEST / "replicate.json", rep)
    print("replicate pass", rep["passed"])


if __name__ == "__main__":
    main()
