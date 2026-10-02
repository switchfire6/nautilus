# Instructions for Claude Code sessions in this repository

**The project.** Nautilus: does a sleep-like offline phase help a learning
machine have insights, and is insight a form of compression? Read
`docs/PROJECT_BRIEF.md` and `docs/related_work.md` before doing anything
substantial. The brief's stages (S0–S4) and method rules (Section 6) are
binding.

**The owner.** A curious, non-specialist researcher who is cost-conscious
about tokens.

- Explain results in plain language first, then technically.
- Prefer lean, direct work (small scripts, few or no subagents) over large
  multi-agent pipelines.
- Ask before starting anything long or expensive.
- Once the owner approves a plan, carry it out without intermediate check-ins,
  and report at the end. Stop at the agreed ✋ gates and for genuinely new
  decisions.
- Give one clear recommendation, not a menu.

**Machine.** A Linux mini PC (Beelink, Intel N150, 4 cores, 15 GiB RAM).
Everything is CPU-only.

- Use a Python virtual environment in `.venv/`.
- Up to 4 parallel workers with 1 thread each is fine; the machine is
  otherwise idle.
- Speed notes from Planaria:
  - call `torch.set_flush_denormal(True)`, because denormal arithmetic made
    runs up to 7× slower;
  - use `SGD(..., foreach=False)` for tiny models;
  - warm up before timing anything;
  - run `pytest` with `OMP_NUM_THREADS=1`, and not at `nice` while a sweep is
    running.
- Launch sweeps longer than 2 hours detached (`setsid nohup`), because tool
  calls time out.

**Research discipline.**

- **Protocols come first.** Write a prospective protocol in `docs/protocols/`
  before any experiment whose result will be claimed. It declares the
  development knobs, development seeds, fresh test seeds, criteria (at most
  5), validity checks, and what each outcome implies.
- **Lock before testing.** Record config, protocol, source and
  analysis-script hashes before a test run. Make the lock a commit on a
  branch. Amend only before the lock, and say so in the protocol. Disclose
  any post-lock deviation.
- **Never tune on test seeds.** Never relax a threshold after seeing data.
  Report negative results as clearly as positive ones.
- **Calibrate bars** against a ceiling (a learner told the shortcut) and a
  floor (the baseline snapshot).
- **Use the strongest fair competitor.** Awake training gets the same compute
  and its own tuning budget. Both predecessor projects lost to simple tuned
  baselines, so assume that will happen and design for it.
- **Replicate.** Run a replicate pass on a subset of runs before analysis.
  Compare results (accuracy, measures), not metadata such as file paths. Hash
  result files.
- **Fractals.** Use fractal or self-similarity ideas only in precise, testable
  roles, each with a non-fractal control. This field attracts pseudo-science.

**Engineering.**

- Keep runs deterministic: fixed seeds,
  `torch.use_deterministic_algorithms` where practical, one thread per worker.
- Make long runs resumable. Raw outputs go in `runs/` (git-ignored); commit
  only summaries, tables and figures under `reports/`.
- Add tests for every task, learner phase and measure. Keep `pytest` and
  `ruff check` passing.

**Git.**

- Commit only when the owner asks, or at clearly agreed checkpoints.
- Use a branch and a pull request for substantial changes.
- Never commit secrets or personal details.
- Commit author: `switchfire6 <switchfire6@users.noreply.github.com>`
  (GitHub's no-reply address, which keeps the owner's email private).
