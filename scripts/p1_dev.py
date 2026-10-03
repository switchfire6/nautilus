"""P1 development phase (protocol §4). Development seeds only.

A. Rung-2 difficulty: baseline switcher share vs mirror exposure, for gate L1 0.01 and
   0.03; choose the design whose share is closest to 40% within 20-60%.
A2. Offline budget (amendment 9): the budget at which plain replay's switcher share is
   closest to the midpoint between baseline and ceiling.
B. Tuning: baseline + 32 awake + 32 sleep configurations on both rungs (test learners and
   control learners); pick the best awake and best sleep arm by switcher share.
C. Dev validity: ablations and ceiling at the selected sleep arm.

Writes configs/p1_design.json, configs/p1_selected.json and reports/P1_dev.json.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/p1_dev.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from nautilus import p1
from nautilus.harness import RUNS_DIR, cached, run_parallel, save_json
from nautilus.tasks import REPO_ROOT

OUT = RUNS_DIR / "p1" / "dev"
N_R1_TEST, N_R1_CTRL, N_R2 = 5, 10, 24
CAL_L1 = (0.01, 0.03)
CAL_MIRROR = 6000
CAL_TOTALS = (2000, 2500, 3000, 4000, 5000, 6000)
M1 = 1000
BUDGETS = {"r1": (10, 20, 50, 100, 200), "r2": (25, 50, 100, 200, 500)}


def r1_job(j):
    return cached(OUT / f"r1_{j['stage']}_{'ctrl' if j['control'] else 'test'}_{j['i']}.json",
                  lambda: p1.r1_batch(("P1", "dev", "r1", j["control"], j["i"]),
                                      j["control"], j["arms"], base_steps=j.get("steps")))


def r2_job(j):
    design = p1.R2Design(**j["design"])
    return cached(OUT / f"r2_{j['stage']}_{'ctrl' if j['control'] else 'test'}_{j['i']}.json",
                  lambda: p1.r2_learner(("P1", "dev", "r2", j["control"], j["i"]),
                                        j["control"], j["arms"], design,
                                        base_steps=j.get("steps")))


def cal_job(j):
    return cached(OUT / f"cal_l1={j['l1']}_{j['i']}.json",
                  lambda: dict(ec=p1.r2_calibration_run(("P1", "dev", "r2", False, j["i"]),
                                                        j["l1"], CAL_MIRROR)))


def any_job(j):
    return {"r1": r1_job, "r2": r2_job, "cal": cal_job}[j["rung"]](j)


def choose_design(cal: dict) -> tuple[dict, dict]:
    table = {}
    for l1, runs in cal.items():
        ec = np.array([r["ec"] for r in runs], float)
        table[l1] = {t: float(np.mean(ec[:, t - 400 : t].mean(axis=1) > p1.R2_MIN_RATE))
                     for t in CAL_TOTALS}
    ok = [(abs(s - 0.4), l1, t) for l1, row in table.items() for t, s in row.items()
          if 0.2 <= s <= 0.6]
    if not ok:
        raise SystemExit(f"no rung-2 design gives 20-60% baseline switchers: {table}")
    _, l1, total = min(ok, key=lambda x: (x[0], x[1], x[2]))
    return dict(m1=M1, m2=total - M1, gate_l1=l1), table


def split(results):
    test = [r for r in results if not r["control"]]
    ctrl = [r for r in results if r["control"]]
    return test, ctrl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.perf_counter()

    # A. rung-2 difficulty
    cal_jobs = [dict(rung="cal", l1=l1, i=i) for l1 in CAL_L1 for i in range(N_R2)]
    cal_res = run_parallel(any_job, cal_jobs, args.workers)
    cal = {l1: [r for j, r in zip(cal_jobs, cal_res) if j["l1"] == l1] for l1 in CAL_L1}
    design, cal_table = choose_design(cal)
    save_json(REPO_ROOT / "configs/p1_design.json", design)
    print("design", design, flush=True)

    # A2. offline budget: plain replay closest to midway between baseline and ceiling
    bjobs = []
    for rung in ("r1", "r2"):
        spec = p1.R1 if rung == "r1" else p1.R2
        arms = [p1.BASELINE, p1.CEILING, p1.plain_replay(spec)]
        for steps in BUDGETS[rung]:
            if rung == "r1":
                bjobs += [dict(rung="r1", stage=f"budget{steps}", control=c, i=i, arms=arms,
                               steps=steps)
                          for c, n in ((False, N_R1_TEST), (True, N_R1_CTRL))
                          for i in range(n)]
            else:
                bjobs += [dict(rung="r2", stage=f"budget{steps}", control=c, i=i, arms=arms,
                               steps=steps, design=design)
                          for i in range(N_R2) for c in (False, True)]
    bres = run_parallel(any_job, bjobs, args.workers)
    budget, budget_table = {}, {}
    for rung, collect, floor in (("r1", p1.r1_collect, p1.R1_MIN_GAIN),
                                 ("r2", p1.r2_collect, p1.R2_MIN_RATE)):
        rows = {}
        for steps in BUDGETS[rung]:
            test, ctrl = split([r for j, r in zip(bjobs, bres)
                                if j["rung"] == rung and j["steps"] == steps])
            sh = p1.shares(collect(test), collect(ctrl), floor)
            rows[steps] = {a: v["share"] for a, v in sh.items()}
        budget_table[rung] = rows
        # baseline and ceiling do not depend on the budget; average over budgets
        base = np.mean([r["baseline"] for r in rows.values()])
        ceil = np.mean([r["ceiling"] for r in rows.values()])
        target = (base + ceil) / 2
        budget[rung] = min(BUDGETS[rung],
                           key=lambda st: (abs(rows[st]["plain_replay"] - target), st))
        print(rung, "budget table", rows, "target", round(target, 3), "chosen", budget[rung],
              flush=True)
    design = dict(design, base_steps=budget)
    save_json(REPO_ROOT / "configs/p1_design.json", design)

    # B. tuning
    grids = {"r1": [p1.BASELINE] + p1.awake_grid(p1.R1) + p1.sleep_grid(p1.R1),
             "r2": [p1.BASELINE] + p1.awake_grid(p1.R2) + p1.sleep_grid(p1.R2)}
    r2_design = {k: v for k, v in design.items() if k != "base_steps"}
    jobs = [dict(rung="r2", stage=f"tune_b{budget['r2']}", control=c, i=i, arms=grids["r2"],
                 design=r2_design, steps=budget["r2"])
            for i in range(N_R2) for c in (False, True)]
    jobs += [dict(rung="r1", stage=f"tune_b{budget['r1']}", control=c, i=i, arms=grids["r1"],
                  steps=budget["r1"])
             for c, n in ((False, N_R1_TEST), (True, N_R1_CTRL)) for i in range(n)]
    res = run_parallel(any_job, jobs, args.workers)
    tuned, selected = {}, {}
    for rung, collect, floor in (("r1", p1.r1_collect, p1.R1_MIN_GAIN),
                                 ("r2", p1.r2_collect, p1.R2_MIN_RATE)):
        test, ctrl = split([r for j, r in zip(jobs, res) if j["rung"] == rung])
        sh = p1.shares(collect(test), collect(ctrl), floor)
        tuned[rung] = sh
        arms = {a["name"]: a for a in grids[rung]}
        best = {}
        for kind in ("awake", "sleep"):  # highest share; ties -> first in grid order
            names = [a["name"] for a in grids[rung] if a["kind"] == kind]
            best[kind] = arms[max(names, key=lambda n: (sh[n]["share"], -names.index(n)))]
        selected[rung] = best
        print(rung, "baseline", sh["baseline"], "best awake", best["awake"]["name"],
              sh[best["awake"]["name"]], "best sleep", best["sleep"]["name"],
              sh[best["sleep"]["name"]], flush=True)
    save_json(REPO_ROOT / "configs/p1_selected.json", selected)

    # C. dev validity: ablations + ceiling at the selected sleep arm
    vjobs = []
    for rung in ("r1", "r2"):
        arms = [p1.BASELINE, p1.CEILING] + p1.ablation_arms(selected[rung]["sleep"])
        if rung == "r1":
            vjobs += [dict(rung="r1", stage=f"valid_b{budget['r1']}", control=c, i=i,
                           arms=arms, steps=budget["r1"])
                      for c, n in ((False, N_R1_TEST), (True, N_R1_CTRL)) for i in range(n)]
        else:
            vjobs += [dict(rung="r2", stage=f"valid_b{budget['r2']}", control=c, i=i,
                           arms=arms, design=r2_design, steps=budget["r2"])
                      for i in range(N_R2) for c in (False, True)]
    vres = run_parallel(any_job, vjobs, args.workers)
    valid = {}
    for rung, collect, floor in (("r1", p1.r1_collect, p1.R1_MIN_GAIN),
                                 ("r2", p1.r2_collect, p1.R2_MIN_RATE)):
        test, ctrl = split([r for j, r in zip(vjobs, vres) if j["rung"] == rung])
        valid[rung] = p1.shares(collect(test), collect(ctrl), floor)
        print(rung, "validity", json.dumps(valid[rung]), flush=True)
    r1_test, _ = split([r for j, r in zip(jobs, res) if j["rung"] == "r1"])
    r2_test, _ = split([r for j, r in zip(jobs, res) if j["rung"] == "r2"])
    report = dict(
        design=design, cal_table=cal_table, budget_table=budget_table, selected=selected,
        tuned=tuned, valid=valid,
        competence=dict(
            r1_easy_end_motion=float(np.nanmean([b["easy_end_motion"] for b in r1_test])),
            r2_step_acc_min=float(min(x["step_acc"] for x in r2_test))),
        r2_design_defaults=asdict(p1.R2Design()),
        wall_s=time.perf_counter() - t0,
    )
    save_json(REPO_ROOT / "reports/P1_dev.json", report)
    print(f"done in {report['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
