"""Record the P1 lock: SHA-256 of the protocol, configs, source and analysis scripts.

    .venv/bin/python scripts/p1_lock.py   # then commit docs/protocols/P1_lock.json
"""

from __future__ import annotations

import subprocess
from datetime import date

from nautilus.harness import file_sha256, save_json
from nautilus.tasks import REPO_ROOT

FILES = (
    ["docs/protocols/P1_sleep_vs_awake.md", "configs/p1_design.json",
     "configs/p1_selected.json", "configs/lowe2024_fitted_motion_means.csv",
     "requirements.txt", "scripts/p1_test.py", "scripts/p1_analyse.py"]
    + sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "src/nautilus").glob("*.py"))
)


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()
    lock = dict(date=str(date.today()), parent_commit=head,
                sha256={f: file_sha256(REPO_ROOT / f) for f in FILES})
    save_json(REPO_ROOT / "docs/protocols/P1_lock.json", lock)
    print(f"locked {len(FILES)} files at parent {head[:10]}")


if __name__ == "__main__":
    main()
