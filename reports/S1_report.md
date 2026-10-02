# S1 report: engineering and the rung-1 replication

*Written 2026-10-02. S1 is engineering, not a claim, and no protocol is locked yet.
The numbers here are calibration for the first protocol (S2). Raw outputs are in
`runs/` (git-ignored); the summaries are `reports/rung1_replication.json`,
`reports/rung2_characterisation.json` and `reports/timings.json`.*

## 1. Plain-language summary

- **We reproduced Löwe et al.'s tiny "insight" network.** With the authors' own
  update rule and analysis procedure, about **40 of 99** networks have an "aha"
  switch (published: 46 of 99). The switch comes about **156 trials** after the
  hidden colour rule starts (published: 175). The "silent knowledge" effect
  reproduces almost exactly: before switching, insight networks already carry about
  twice as much hidden colour weight. More noise gives more insight networks, and a
  stronger regulariser gives fewer and later ones, as published.
- **The paper's text is not enough to reproduce it.** Built from the equations
  printed in the paper, about **95%** of networks switch. The published numbers come
  from details found only in the authors' code, for example how noise and
  regularisation are scaled, and 0/1 rather than ±1 targets.
- **The insight count is fragile.** A network counts as "insight" if its jump beats
  the single most extreme of 99 control networks. That one control network moves the
  count between 21 and 50 across repeat batches with identical settings. The count
  also depends on the curve-fitting routine. The authors' fitter barely adjusts the
  slope from its starting value. A best-fit version gives about 22 of 99 instead of
  40. **For our own experiments we should use a sturdier insight measure.**
- **Rung 2 (the number puzzle) works mechanically.** A small recurrent network learns
  the step rules, then some learners start answering at step 3 instead of step 8,
  with accuracy unchanged. The default settings make this too common (11 of 12)
  for a fair test, so the task needs tuning in S2.
- **Compute accounting caught a real problem.** With equal update steps, the awake
  arm uses 1.3–1.5× the arithmetic of sleep. "Equal steps" and "equal FLOPs" can't
  both hold as built, so S2 must choose which to equalise.

## 2. Rung 1: replication of Löwe et al. (2024)

### 2.1 What we implemented

- **Source of truth.** The authors' code
  ([gitlab.com/aloewe/insightnets](https://gitlab.com/aloewe/insightnets)), including
  their 99 fitted per-network motion means (`configs/lowe2024_fitted_motion_means.csv`).
- **Fitting.** The analysis uses their sigmoid classification, reproduced from
  `SwitchNetsBehavior.R`, with the same NLopt COBYLA optimiser, start point and
  tolerances.
- **Trial sequences.** We generate our own sequences with the statistics of theirs:
  per 100 trials, 30/10/20/20/20 trials at the five coherence levels, balanced
  answers.

**Where the paper text and the code differ** (our default follows the code):

| Detail | Paper text | Authors' code |
|---|---|---|
| Targets | y = ±1 | y ∈ {0, 1} |
| Colour when not predictive | ±M_c at random | mean 0 (noise only) |
| Pre-training | 6 blocks | 8 blocks |
| Update noise | +ξ | +α·ξ |
| L1 step on gates | −αλ·sign(g) | −λ·sign(g), and a gate that crosses 0 is set to 0 |
| Output noise σ_η | not stated | 0.3 |
| Motion input SD | "0.01" and "0.1" in different places | 0.1 |
| Initial parameters | not stated | all 0.01 |

### 2.2 Results (10 batches × 99 test + 99 control networks)

| Quantity | Published | Ours (authors' fit) | Ours (best fit) |
|---|---|---|---|
| Insight networks / 99 | 46 | **39.8 ± 9.2** (range 21–50) | 22.0 ± 10.9 |
| Switch delay (trials after colour onset) | 175 ± 105 | **156 ± 99** | 146 ± 111 |
| \|w_c\| at colour onset, insight vs no-insight | 1.2 vs 0.4 | **1.13 vs 0.52** | 1.19 vs 0.64 |
| \|w_m\| at colour onset | 3.4 vs 3.5 | 3.14 vs 3.38 | |
| Colour-gate gradient, first 5 colour trials | 0.05 vs 0.01 | 0.096 vs 0.040 | |
| Gates at end, g_c (insight vs no) | 0.7 vs 0.07 | 0.67 vs 0.16 | |
| Gates at end, g_m (insight vs no) | 0.2 vs 0.5 | 0.13 vs 0.44 | |
| Hardest-level accuracy, motion phase | 58% vs 64% | 58% vs 62% | |
| Hardest-level accuracy, colour phase | 83% vs 64% | 83% vs 65% | |

Two notes on these rows:

- **Gradient row.** The paper doesn't define its gradient measure. Ours is
  |∂L/∂g_c|, so only the direction and ratio are comparable.
- **Unexpected detail.** The paper's "unexpected" finding also reproduces: insight
  networks are slightly *worse* before the switch.

**Paper-text variant:** 95.3 ± 1.2 of 99 (3 batches). **Six pre-training blocks with
the code's rule:** 44.0 ± 5.6, so this detail doesn't matter.

**Noise and regularisation sweeps** (5 batches per level, authors' fit, each level
with its own controls):

| σ_ξ | 0 | 0.01 | 0.03 | 0.05 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| insight / 99 | 0 | 3.4 | 17.6 | 41.2 | 77.6 | 88.8 | 82.2 |

| λ | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 |
|---|---|---|---|---|---|---|---|
| insight / 99 | 86.2 | 79.6 | 61.2 | 42.8 | 49.6 | 44.4 | 38.4 |
| delay (trials) | 116 | 139 | 140 | 128 | 151 | 154 | 167 |

- **Qualitative match.** Without noise there is no insight. Insight rises with
  noise, falls with λ, and comes later with larger λ.
- **Quantitative difference.** The paper reports a ceiling of about 100% near
  σ_ξ = 0.05. We see 41% there and a plateau near 85–90% at σ_ξ ≥ 0.1.
- **A likely cause.** The authors' σ_ξ-sweep notebook (`xi_avg.ipynb`) writes the
  accuracy *averaged over all 99 networks* into every network's row
  (`av_acc[crun,:,:] = np.mean(acc,0)`). Classification is then all-or-none per
  run. If their Fig. 5 came from these files, it shows the share of runs whose
  *average* curve beat the controls, not the share of networks. We count networks
  individually.
- **Divergence.** At σ_ξ = 0.5, 7 of 990 networks diverged. They count as no
  insight.

### 2.3 Issues we found in the insight measure

1. **The threshold is one extreme value.** A network is "insight" if its score
   beats the *maximum* of 99 controls. Across our 10 main batches that maximum
   ranged from 0.09 to 0.66, and the insight count followed it, from 50 down to 21.
   The batch with 21 had one control network with an unusually large chance jump.
2. **The slope is not really fitted.** COBYLA from the authors' start (slope 7)
   ends at slope 7.2 on average for *both* groups. The "steepness" score is
   therefore mostly the jump size times a constant. The true least-squares fit
   gives different slopes and about half as many insight networks.
3. **Consequence for S2.** We should pre-declare a sturdier insight criterion that
   doesn't depend on one control draw or one optimiser start. Examples: a
   multi-start fit, a threshold at a high quantile of a *large* control pool, or a
   direct jump-size criterion. We should also report the Löwe criterion alongside it
   for comparability.

### 2.4 Integrity

- **Determinism.** Seeds are derived from job names. A replicate pass recomputed 3
  batches from scratch, and every insight label and score was bit-identical
  (`scripts/replicate_check_rung1.py`).
- **Hashes.** The summary JSON is hashed when written. The fitted-means file hash is
  recorded in the summary.

## 3. Rung 2: Number Reduction Task learner

- **Task.**
  - Strings of 8 digits from {1, 4, 9}, reduced left to right.
  - Mirror strings have the response pattern r1, A, B, C, C, B, A, so the 2nd response
    equals the final answer.
  - Coded as 0/1/2, the rule is **modular addition**, r = −(a+b) mod 3, which is
    the classic "grokking" function.
- **Learner.**
  - A GRU with 32 hidden units reads (digit, position).
  - One head gives the current chained response.
  - A second head, behind L1-regularised gates, gives the final answer at every step.
  - The learner answers at the first step where its confidence reaches 0.9.
  - **Insight** means answering at step ≤ 3 and being correct.
- **Curriculum.**
  - **Instruction:** Adam teaches the step rules. Plain SGD stays at chance on
    modular addition for over 16,000 strings, and humans are told the rules anyway.
  - **Pre-phase:** 2,000 unstructured strings with online SGD.
  - **Main phase:** 6,000 strings, mirror strings for test learners and
    unstructured strings for controls.
- **Results (12 mirror + 12 control learners, defaults).**
  - **Rules learned:** step-rule accuracy is 99.9%.
  - **Switching:** 11 of 12 mirror learners switch (9 of 12 answer early on more
    than half the trials by the end). No control ever answers early.
  - **Timing:** switches come 2,500–5,800 strings after mirror onset and are
    fairly abrupt (early-and-correct 0 → 0.9 within about 2,000 strings in the
    clearest case).
- **Not yet usable for H1.**
  - **No headroom:** insight is near 100%.
  - **Degenerate threshold:** controls all score exactly 0, so the threshold
    means nothing.
  - **Fix in S2:** tune the main-phase length, L1 strength or noise on development
    seeds to put insight near 30–60%, and add a minimum-effect criterion.
- **Compression measures.**
  - **Over the main phase:** TwoNN intrinsic dimension and correlation dimension of
    the step-3 hidden states fall (TwoNN 5.2 → 4.6), while effective rank *rises*
    (13.1 → 14.6).
  - **Controls:** they drift less (TwoNN 5.0 → 4.8) and their effective rank stays flat.
  - **Lead times:** naive change-point lead times all come out negative ("compression
    first"). That is **not evidence yet**: it ignores control drift, and the
    change-point method may favour early onsets. H2 needs control-calibrated change
    points, declared in a protocol.

## 4. Phases and compute accounting

- **Awake.** One stage, repeated: each step replays stored experiences, with the
  learner's continuous regulariser and noise. Both are tunable.
- **Sleep.** Cycles of NREM-like steps (replay plus scaling-down) and REM-like steps
  (noise, no replay). No new data is used; tests check that the buffer is the only
  data source.
- **Ablations.** Replay only, noise only, or shrink only. Each has the same step
  count and the same total noise variance or total shrink as the full sleep phase.
- **Budget check.** `check_equal_budget` compares update steps and analytic FLOPs.

**Timings** (one core; awake and sleep phases run 1,000 steps):

| Item | Time |
|---|---|
| Rung 1: simulate 99 networks through the full curriculum | 0.17 s |
| Rung 1: one replication batch (99 + 99 networks, two fit methods) | ≈ 47 s (fits dominate) |
| Rung 1: all 86 batches, 4 workers | 17 min |
| Rung 1 phases (99 networks): awake / sleep step | 0.12 / 0.08 ms |
| Rung 2: one learner (instruction + 8,000 strings + probes) | 7.6 s |
| Rung 2 phases (1 learner, batch 8): awake / sleep step | 2.8 / 2.1 ms |

**FLOPs check.** With equal steps, awake/sleep FLOPs are 1.5× (rung 1) and 1.3×
(rung 2). The verification flags this as unequal. S2 has to choose a matching rule:

- equal FLOPs, which gives sleep more steps;
- or REM steps that also cost a replay gradient, for example "dream" replay on
  perturbed inputs.

## 5. Open decisions for S2

1. The insight criterion (§2.3).
2. Rung-2 difficulty settings to restore headroom.
3. The compute-matching rule (§4).
4. Where the offline phase sits in the curriculum. Sleep could come between the
   motion and colour phases, or part-way into the colour phase, as in Wagner's
   design.
5. A control-calibrated change-point method for lead times.
