# Protocol P2: does internal compression come first and predict insight? (H2, rung 2)

*Status: **v1**, 2026-10-02. Not locked. The owner delegated the design. Amendments
are allowed until the lock and will be listed in §9.*

## 1. Question

Rung-2 learners see the mirror strings with **no offline phase**. Some of them later
switch to answering early.

- Does a change in their internal representation during the **first 1,000 mirror
  strings**, before anyone switches, predict *which* learners will switch?
- Does it predict better than the strongest non-compression signal available?

The "lead" part of H2 is reported descriptively (§6).

## 2. Design (fixed)

- **Learners.** Rung-2 learners exactly as in P1: instruction, then 2,000 unstructured
  strings, then mirror strings, all with online SGD.
- **Mirror phase length.** **5,000** strings, with gate L1 0.01. On P1 development
  seeds, about 54% of learners had switched by this point.
- **Outcome.** A learner is a **switcher** if its early-and-correct rate over the last
  400 strings is above 0.5. This is the P1 rule; P1 controls never answer early, so the
  0.5 floor binds.
- **Probes.** A fixed set of 300 mirror strings per learner. On it we record step-3
  hidden states and the final-answer head's probability of the correct answer at step
  3. Probes are taken every 100 strings up to string 1,000, then every 250.
- **Window.** Predictors use the change between mirror strings 0 and 1,000. In P1
  calibration no development learner had switched by 2,500 strings, so this window is
  well before any switch.

## 3. Predictors (direction declared in advance)

| Predictor | Kind | Score (higher means a switch is predicted) |
|---|---|---|
| **P_TwoNN (primary)** | compression | −Δ TwoNN intrinsic dimension of step-3 states |
| P_corr | compression | −Δ correlation dimension |
| P_rank | compression | −Δ effective rank |
| **K_silent (competitor)** | non-compression ("silent knowledge") | +Δ mean probability of the correct final answer at step 3 (below the answer threshold) |
| K_move (sanity) | non-compression | ‖θ₁₀₀₀ − θ₀‖, how far the weights moved |

## 4. Criteria (at most 5)

- **C1, prediction.** The primary predictor's AUC for switching is at least 0.65, and
  the lower bound of its 95% bootstrap CI is above 0.5.
- **C2, added value over the competitor.** Fit a logistic regression with 10-fold
  stratified cross-validation, once on (K_silent, P_TwoNN) and once on K_silent alone.
  The AUC gain from adding P_TwoNN must have a 95% bootstrap CI lower bound above 0.

**H2's prediction claim** is supported only if both C1 and C2 pass.

## 5. Validity checks and futility rule

- **V1, headroom:** the switcher share is 25–75% on test seeds.
- **V2, competence:** step-rule accuracy above 0.95 at the start of the mirror phase.
- **V3, replicate pass:** 3 learners recomputed from scratch give identical outcomes
  and predictors.
- **Futility (development).** Development seeds: `("P2", "dev", i)`, 48 learners. If
  *no* compression predictor reaches AUC ≥ 0.60 on them, the test phase is **not run**.
  P2 is then reported as a development-stage negative, a weaker statement that is
  flagged as such. There are no development knobs, so development serves only as this
  pilot.

## 6. Descriptive (not criteria)

- **AUCs** of every predictor.
- **Lead time:** for switchers, the hinge onset of the TwoNN series over the whole
  mirror phase minus the behavioural switch bin. This is reported with the S1 caveat
  that the hinge may favour early onsets.

## 7. Test phase

- **Test seeds:** `("P2", "test", i)`, 192 learners.
- **Lock before testing:** hashes of the protocol, `src/` and `scripts/p2_*.py` are
  recorded in a lock commit.

## 8. What each outcome implies

- **C1 and C2 pass.** Early compression carries information about future insight
  beyond silent knowledge. This supports H2's "compression comes first".
- **C1 passes, C2 fails.** Compression predicts insight only as a shadow of silent
  knowledge. This counts as a negative for H2's distinctive claim.
- **C1 fails, or futility.** No evidence that compression precedes or predicts insight
  in this learner. Reported as a negative.

## 9. Amendment log

- *(none)*
