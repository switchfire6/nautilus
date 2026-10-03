# Nautilus

**Does "sleeping on it" help a learning machine have insights?**

People often crack a problem after a night's sleep. In one classic experiment,
more than twice as many people found a hidden shortcut after sleeping. This
project tests the idea in small artificial learners, which run on a CPU:

- Does a sleep-like offline phase (replay, noise, pruning) lead to more
  "aha" moments than the **same amount of computation** spent practising
  awake?
- Is insight a form of **compression**, finding a simpler description that
  connects the dots? Does that compression start before the behaviour changes?
- Does a bias toward **self-similar** (fractal) descriptions help when the
  hidden rule is itself self-similar?

A nautilus shell grows by repeating its own shape at a larger scale: a complex
form from a short rule.

**Status:** S0 (literature check) and S1 (engineering, rung-1 replication) done;
S2 (first protocol) next. See:

- the [project brief](docs/PROJECT_BRIEF.md) for the hypotheses, design and
  research rules;
- [related work](docs/related_work.md) for what is already known, and the gap
  this project tests;
- the [S1 report](reports/S1_report.md) for the replication of Löwe et al. (2024)
  and the engineering checks.

**Predecessors** (same research discipline: pre-registered protocols, strong
baselines, negatives reported):

- [ACP-CL](https://github.com/switchfire6/ACP-CL);
- [Planaria](https://github.com/switchfire6/planaria), whose result was a
  calibrated negative.

## License

[MIT](LICENSE)
