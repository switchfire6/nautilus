"""P2 runs: ``dev`` (pilot + futility check) or ``test`` (requires a matching lock).

    OMP_NUM_THREADS=1 .venv/bin/python scripts/p2_run.py dev|test [--workers 4]
"""

from __future__ import annotations

import argparse
import json

from nautilus import p2
from nautilus.harness import RUNS_DIR, cached, file_sha256, run_parallel, save_json
from nautilus.tasks import REPO_ROOT

N = {"dev": 48, "test": 192}
LOCK = REPO_ROOT / "docs/protocols/P2_lock.json"


def job(j: dict) -> dict:
    return cached(RUNS_DIR / "p2" / j["stage"] / f"{j['i']}.json",
                  lambda: p2.run_learner(("P2", j["stage"], j["i"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dev", "test"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "test":
        lock = json.loads(LOCK.read_text())
        bad = [f for f, h in lock["sha256"].items() if file_sha256(REPO_ROOT / f) != h]
        if bad:
            raise SystemExit(f"lock mismatch: {bad}")
    runs = run_parallel(job, [dict(stage=args.stage, i=i) for i in range(N[args.stage])],
                        args.workers)
    out = p2.analyse(runs)
    if args.stage == "dev":
        out["futile"] = bool(max(out["aucs"][k] for k in ("P_twonn", "P_corr", "P_rank"))
                             < 0.60)
    else:
        again = run_parallel(p2.run_learner, [("P2", "test", i) for i in (7, 70, 140)],
                             args.workers)
        out["V3_replicate"] = all(
            json.dumps(a, sort_keys=True) == json.dumps(runs[i], sort_keys=True)
            for a, i in zip(again, (7, 70, 140)))
    digest = save_json(REPO_ROOT / f"reports/P2_{args.stage}.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "lead_strings"}, indent=1))
    print("sha256", digest)


if __name__ == "__main__":
    main()
