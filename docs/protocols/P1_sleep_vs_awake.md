# Protocol P1: does sleep beat equal awake compute? (H1, rungs 1–2)

*Status: **v1 + amendments 1–9, LOCKED 2026-10-02** (hashes in `P1_lock.json`).
The development phase is complete (`reports/P1_dev.json`); no test-seed run was
made before the lock. Any post-lock deviation is disclosed in the results report,
not edited in here.*

## 1. Question

After a first learning session, does an offline **sleep** phase lead more learners to
discover the hidden shortcut than the **same compute spent practising awake** on the
same stored experiences? (H1 in `docs/PROJECT_BRIEF.md`.) H2 (compression) is measured
descriptively only; it is tested in P2.

## 2. Design (fixed, not tuned)

**Paired design.** Each learner runs session 1 once. A snapshot is then copied into
every arm, so all arms share the same learner and the same session-2 trials. Only the
offline phase differs between arms.

| | Rung 1 (Löwe task, authors'-code variant) | Rung 2 (Number Reduction Task) |
|---|---|---|
| Session 1 | Pre-training, motion phase, then the first 100 colour-phase trials | Instruction, 2,000 strings without structure, then M₁ mirror strings |
| Stored experiences (buffer) | The 100 colour-phase trials of session 1 | The M₁ mirror strings of session 1 |
| Offline budget | 200 update steps (batch 1) | 500 update steps (batch 8) |
| Session 2 | 300 colour-phase trials, native learning rule | M₂ mirror strings, native learning rule |
| Learners per batch | 99 (authors' fitted motion means) | 1 |

**Arms.**

1. **Baseline** (floor): no offline phase.
2. **Awake**: replay on the buffer every step, with continuous regularisation and noise.
   Tuned (§4).
3. **Sleep**: cycles of 8 NREM steps (replay + scaling-down) and 2 REM steps (noise).
   Tuned (§4).
4. **Replay-only**, 5. **noise-only** and 6. **shrink-only** ablations, at the tuned
   sleep strengths (`phases.ablation`).
7. **Ceiling** (a learner told the shortcut):
   - rung 1: the colour gate is set to 0.5 at the start of session 2;
   - rung 2: the instruction stage also uses mirror strings.

**Control learners.** Each arm is also run on control learners, whose shortcut never
becomes valid. These give that arm's own false-positive threshold (§3).

**Compute matching.** Awake and sleep are matched on **counted FLOPs**, within 2%,
with the awake budget as the reference. Sleep therefore gets more steps, and both step
counts are reported. Ablations use sleep's step count; their lower FLOPs are reported,
not matched. (Why: S1 showed that equal steps leave awake with 1.3–1.5× the FLOPs, so
matching steps would favour awake.)

## 3. Insight measure (primary)

**Why not the Löwe criterion.** In S1 it proved fragile. Its threshold is the *maximum*
score of 99 controls, which swung the count between 21 and 50 out of 99 across
identical batches. Its fitted slope barely moves from the optimiser's starting value.
It is kept as a secondary measure, for comparability.

**Rung 1.**

- Gain Δ = (analytic accuracy at the hardest coherence level over the last 100 trials
  of session 2) − (the same over the motion phase).
- A learner is a **switcher** if Δ > max(q₉₉, 0.10). Here q₉₉ is the 99th percentile
  of Δ among that arm's control learners, a pool of at least 990.

**Rung 2.**

- A learner is a **switcher** if its early-and-correct rate over the last 400 strings
  of session 2 is above max(q₉₉, 0.5), with q₉₉ taken from that arm's control learners.

**Outcome per arm:** the share of switchers. Secondary measures are the time to switch
(the first 50-trial or 400-string bin above threshold), the Löwe classification, and
final accuracy.

## 4. Development phase

**Development seeds.** Seed keys `("P1", "dev", rung, i)`:

- rung 1: 5 batches of 99 learners;
- rung 2: 24 learners.

**Development knobs.** Each tuned arm gets the same tuning budget of 32
configurations, and the best configuration is chosen by the primary outcome on
development seeds.

- **Awake:**
  - learning rate {0.3, 0.6};
  - L1 strength on the gates {0, ½, 1, 2}× the native λ;
  - parameter noise {0, ½, 1, 2}× the native per-step noise SD.

  For rung 2 the multipliers apply to its native values instead.
- **Sleep:**
  - learning rate {0.3, 0.6};
  - L2 downscaling per NREM step {0, 0.001, 0.003, 0.01};
  - REM noise SD {0, 1, 2, 4}× the native per-step noise SD.
- **Rung-2 difficulty** (M₁, M₂ and gate L1). Chosen *before* tuning the arms, so that
  the baseline switcher share is 20–60% on development seeds.

## 5. Test phase

- **Fresh test seeds.** Seed keys `("P1", "test", rung, i)`:
  - rung 1: 20 batches of 99 learners;
  - rung 2: 96 learners.
- **Lock before testing.** Before any test run, record the hashes of this protocol, the
  configs, `src/` and the analysis script in a lock commit.

## 6. Criteria (at most 5)

- **C1, rung 1: sleep > awake.** The paired-bootstrap 95% CI of
  (sleep share − awake share) has a lower bound above 0, **and** sleep's share is
  higher in at least 15 of the 20 test batches.
- **C2, rung 2: sleep > awake.** The paired-bootstrap 95% CI has a lower bound above
  0, **and** sleep's share is higher in at least 3 of 4 seed blocks of 24 learners.

Descriptive results, which are not criteria:

- each ablation's share relative to sleep and awake;
- time to switch;
- the Löwe classification;
- compression measures and lead times (rung 2).

## 7. Validity checks (all must hold for the criteria to be read)

- **V1, headroom:** the baseline switcher share is 10–70% on test seeds, per rung.
- **V2, competence:** the learners learned the basic task.
  - Rung 1: accuracy above 0.75 at the easiest coherence level at the end of the motion
    phase.
  - Rung 2: step-rule accuracy above 0.95.
- **V3, ceiling:** the ceiling arm's share is at least 90%.
- **V4, equal compute:** awake and sleep FLOPs are within 2%, verified from the
  ledgers.
- **V5, replicate pass:** a subset of test runs, recomputed, gives identical labels.

## 8. What each outcome implies

- **C1 or C2 passes → GO to S3.** Sleep's arrangement of the ingredients beats the best
  continuous use of them on that rung. The ablations then point to the ingredient
  responsible.
- **Sleep > baseline but not > awake on both rungs → calibrated negative.** "Sleep is
  extra compute." It is reported at full detail, and is the most likely outcome given
  the predecessor projects.
- **Awake > sleep → reported as such.**
- **A validity check fails → that rung is not interpretable.** We report why. We do not
  retune on test seeds.

## 9. Analysis

- Paired bootstrap over learners: 10,000 resamples. For rung 1, the resampling unit is
  the batch.
- The analysis script is written and hashed before the lock.

## 10. Amendment log (all made before the lock, while building the code)

1. **Rung-1 ceiling.**
   - **Problem with v1.** Setting the colour gate to 0.5 alone did not work: the
     native L1 step (0.07 per trial) closes the gate within a few trials.
   - **Change.** The ceiling now opens the colour gate (0.5, with a positive colour
     weight) and **holds both gates fixed during session 2**. This is how the
     authors' own code models instructed participants: in their "instructed" block
     the gates are frozen and the weights keep learning.
2. **Rung-2 ceiling.**
   - **Problem with v1.** Using mirror strings in the instruction stage would be
     erased by the 2,000 unstructured practice strings that follow.
   - **Change.** At the start of session 2 the learner is instead *taught* on the
     stored strings: 200 Adam updates, batch 32. Not compute-matched; it is a
     calibration arm.
3. **Rung-2 tuning scales.** The native rung-2 learner has no update noise. Changes:
   - the noise unit is set to 0.003 per step;
   - the learning-rate options are {0.05, 0.1}, i.e. ½× and 1× the native 0.1;
   - the L1 unit is the native per-step gate shrink, 0.1 × 0.01.
4. **FLOP budget definition.**
   - **Budget.** It is the cost of 200 (rung 1) or 500 (rung 2) *full* awake steps
     (replay + noise + shrink).
   - **Matching.** Every awake and sleep configuration is fitted to it within 2%
     (`phases.fit_to_budget`). Awake configurations with an ingredient switched off
     therefore get more replay steps.
5. **Rung-2 difficulty procedure.** M₁ = 1,000 is fixed. Two things are chosen from
   baseline-only development runs (24 learners):
   - the gate L1 strength, from {0.01, 0.03};
   - the total mirror exposure M₁ + M₂, from {2,000, 2,500, 3,000, 4,000, 5,000,
     6,000}.

   The rule is to pick the baseline share closest to 40% within 20–60%; ties go to
   0.01, then the shorter exposure.
6. **Control pools.**
   - Rung-2 control learners: 24 per arm on development seeds and 96 on test seeds.
   - Rung-1 control batches: 10 (990 networks) on development seeds and 20 (1,980)
     on test seeds.
7. **Tie-break in tuning.** Among equal development shares, the configuration that
   comes first in grid order wins.
8. **Secondary measures.**
   - **Time to switch:** rung-1 bins of 50 trials, rung-2 bins of 500 strings (400 does
     not divide M₂ = 3,000; the final-rate window stays the last 400 strings).
   - **Löwe criterion:** rung 1 only, for the baseline, awake and sleep arms.
   - **H2 (descriptive):** the change in TwoNN and effective rank of step-3 hidden
     states over session 2, for the baseline, awake and sleep arms.
9. **Offline budget recalibrated on development seeds.**
   - **What we saw.** v1 fixed the budget at 200 (rung 1) and 500 (rung 2) full awake
     steps. At that budget, plain replay alone made about 96% (rung 1) and 100%
     (rung 2) of development learners switch. The best awake and best sleep arms were
     both at ceiling (95.6% vs 95.8%; 100% vs 100%), so the test could not
     discriminate between them.
   - **Change.** The budget is now chosen on development seeds, from
     {10, 20, 50, 100, 200} (rung 1) or {25, 50, 100, 200, 500} (rung 2) full awake
     steps. The rule: pick the budget at which **plain replay** (awake at the native
     learning rate, no added L1 or noise) has a switcher share closest to the midpoint
     between baseline and ceiling. The rule uses only the awake reference arm, never
     sleep.
   - **Then.** Both grids are re-tuned at the chosen budget. The superseded
     development runs are kept in `runs/p1/dev_superseded/`.
   - **Other development observation at the old budget.** Rung-2 sleep with L2
     downscaling of 0.003 or more per NREM step collapsed the learners (0% switch).
