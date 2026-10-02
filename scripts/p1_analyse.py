"""P1 analysis of the test runs (protocol §6-9). Written and hashed before the lock.

    OMP_NUM_THREADS=1 .venv/bin/python scripts/p1_analyse.py
"""

from __future__ import annotations

import json

import numpy as np

from nautilus import p1
from nautilus.harness import RUNS_DIR, load_json, save_json
from nautilus.measures import classify_insight, fit_switch
from nautilus.tasks import REPO_ROOT

TEST = RUNS_DIR / "p1" / "test"
N_R1, N_R2, R2_BLOCK = 20, 96, 24


def load(rung: str, control: bool, n: int) -> list[dict]:
    tag = "ctrl" if control else "test"
    return [load_json(TEST / f"{rung}_{tag}_{i}.json") for i in range(n)]


def first_bin_above(bins: np.ndarray, thr: float) -> np.ndarray:
    """Index of the first bin above ``thr`` (NaN if none)."""
    above = np.asarray(bins) > thr
    idx = np.argmax(above, axis=-1).astype(float)
    idx[~above.any(axis=-1)] = np.nan
    return idx


def analyse_r1(sel: dict) -> dict:
    test, ctrl = load("r1", False, N_R1), load("r1", True, N_R1)
    dt, dc = p1.r1_collect(test), p1.r1_collect(ctrl)
    sh = p1.shares(dt, dc, p1.R1_MIN_GAIN)
    aw, sl = sel["awake"]["name"], sel["sleep"]["name"]
    per_batch = {a: np.mean(p1.switchers(dt[a], sh[a]["threshold"]), axis=1) for a in dt}
    ci = p1.paired_bootstrap(per_batch[sl], per_batch[aw], seed=1)
    wins = int(np.sum(per_batch[sl] > per_batch[aw]))
    c1 = dict(ci=ci, sleep_wins=wins, n_batches=N_R1,
              passed=bool(ci["lo"] > 0 and wins >= 15))

    # secondary: time to switch (session-2 bins of 50 trials) and the Löwe criterion
    timing = {}
    for a in dt:
        bins = np.array([b["results"][a]["bin_gain"] for b in test], float)
        idx = first_bin_above(bins, sh[a]["threshold"])
        timing[a] = float(np.nanmean(idx)) if np.isfinite(idx).any() else None
    lowe = {}
    for a in ("baseline", aw, sl):
        st = np.concatenate([np.array(b["results"][a]["series"]) for b in test])
        sc = np.concatenate([np.array(b["results"][a]["series"]) for b in ctrl])
        fs = [fit_switch(s, 4, bounds=((0.35, 1.0), (0.0, 12.0), (0.0, 9.0))).score for s in st]
        fc = [fit_switch(s, 4, bounds=((0.35, 1.0), (0.0, 12.0), (0.0, 9.0))).score for s in sc]
        lowe[a] = float(np.mean(classify_insight(fs, fc)))
    ledgers = {a: [b["results"][a]["ledger"] for b in test + ctrl] for a in (aw, sl)}
    flops_ok = all(abs(x["flops"] - x["budget"]) <= 0.02 * x["budget"]
                   for v in ledgers.values() for x in v)
    return dict(
        shares=sh, per_batch={a: v.tolist() for a, v in per_batch.items()}, C1=c1,
        sleep_vs_baseline=p1.paired_bootstrap(per_batch[sl], per_batch["baseline"], seed=2),
        awake_vs_baseline=p1.paired_bootstrap(per_batch[aw], per_batch["baseline"], seed=3),
        first_switch_bin=timing, lowe_share=lowe,
        steps={a: ledgers[a][0]["steps"] for a in ledgers},
        validity=dict(
            V1_headroom=bool(0.10 <= sh["baseline"]["share"] <= 0.70),
            V2_competence=bool(np.nanmean([b["easy_end_motion"] for b in test]) > 0.75),
            V3_ceiling=bool(sh["ceiling"]["share"] >= 0.90),
            V4_equal_flops=bool(flops_ok)),
        competence_easy_end_motion=float(np.nanmean([b["easy_end_motion"] for b in test])),
    )


def analyse_r2(sel: dict) -> dict:
    test, ctrl = load("r2", False, N_R2), load("r2", True, N_R2)
    ft, fc = p1.r2_collect(test), p1.r2_collect(ctrl)
    sh = p1.shares(ft, fc, p1.R2_MIN_RATE)
    aw, sl = sel["awake"]["name"], sel["sleep"]["name"]
    sw = {a: p1.switchers(ft[a], sh[a]["threshold"]).astype(float) for a in ft}
    ci = p1.paired_bootstrap(sw[sl], sw[aw], seed=4)
    blocks = [(sw[sl][k : k + R2_BLOCK].mean(), sw[aw][k : k + R2_BLOCK].mean())
              for k in range(0, N_R2, R2_BLOCK)]
    wins = int(sum(s > a for s, a in blocks))
    c2 = dict(ci=ci, block_shares=blocks, sleep_block_wins=wins,
              passed=bool(ci["lo"] > 0 and wins >= 3))
    timing = {}
    for a in ft:
        bins = np.array([x["results"][a]["bins"] for x in test], float)
        idx = first_bin_above(bins, sh[a]["threshold"])
        timing[a] = float(np.nanmean(idx)) if np.isfinite(idx).any() else None

    # H2, descriptive: change of step-3 hidden-state measures over session 2
    h2 = {}
    for a in ("baseline", aw, sl):
        rows = []
        for x, s in zip(test, sw[a]):
            pr = x["results"][a]["probes"]
            rows.append((s, pr[-1]["twonn"] - pr[0]["twonn"], pr[-1]["erank"] - pr[0]["erank"]))
        r = np.array(rows, float)
        h2[a] = {g: dict(d_twonn=float(np.nanmean(r[m, 1])), d_erank=float(np.nanmean(r[m, 2])),
                         n=int(m.sum()))
                 for g, m in (("switchers", r[:, 0] == 1), ("non_switchers", r[:, 0] == 0))}
    ledgers = {a: [x["results"][a]["ledger"] for x in test + ctrl] for a in (aw, sl)}
    flops_ok = all(abs(x["flops"] - x["budget"]) <= 0.02 * x["budget"]
                   for v in ledgers.values() for x in v)
    return dict(
        shares=sh, C2=c2,
        sleep_vs_baseline=p1.paired_bootstrap(sw[sl], sw["baseline"], seed=5),
        awake_vs_baseline=p1.paired_bootstrap(sw[aw], sw["baseline"], seed=6),
        first_switch_bin=timing, h2_descriptive=h2,
        steps={a: ledgers[a][0]["steps"] for a in ledgers},
        validity=dict(
            V1_headroom=bool(0.10 <= sh["baseline"]["share"] <= 0.70),
            V2_competence=bool(min(x["step_acc"] for x in test) > 0.95),
            V3_ceiling=bool(sh["ceiling"]["share"] >= 0.90),
            V4_equal_flops=bool(flops_ok)),
        competence_step_acc_min=float(min(x["step_acc"] for x in test)),
    )


def main() -> None:
    sel = json.loads((REPO_ROOT / "configs/p1_selected.json").read_text())
    rep_path = TEST / "replicate.json"
    out = dict(r1=analyse_r1(sel["r1"]), r2=analyse_r2(sel["r2"]),
               V5_replicate=load_json(rep_path) if rep_path.exists() else None)
    digest = save_json(REPO_ROOT / "reports/P1_results.json", out)
    print(f"reports/P1_results.json sha256 {digest}")
    for rung, c in (("r1", "C1"), ("r2", "C2")):
        r = out[rung]
        print(rung, c, json.dumps(r[c]), "validity", r["validity"])
        print("  shares", {a: round(v["share"], 3) for a, v in r["shares"].items()})


if __name__ == "__main__":
    main()
