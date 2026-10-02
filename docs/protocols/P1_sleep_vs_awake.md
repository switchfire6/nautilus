# Protocol P1: does sleep beat equal awake compute? (H1, rungs 1–2)

*Status: **v1, approved by the owner 2026-10-02**. Not locked. Amendments are allowed until the lock
commit; every amendment made before the lock will be listed in §10. No development or
test runs have been made under this protocol.*

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

## 10. Amendment log

- *(none yet)*
