# P2 result: development-stage negative (futility rule triggered)

*Protocol: `docs/protocols/P2_compression_predicts.md` (v1). Development seeds only:
48 learners. Summary: `reports/P2_dev.json`. Per the pre-declared futility rule (§5),
the test phase was **not run** and the protocol was not locked. This is a weaker
statement than a test-seed negative.*

## Plain language

- **The question.** We watched each learner's internal "thought space" during the first
  1,000 mirror strings, well before anyone switched. Did its compression (shrinking
  intrinsic dimension, correlation dimension or effective rank) predict which learners
  would later have the insight?
- **The answer is no.** Every compression measure did no better than a coin flip. The
  best AUC was 0.51, where 0.5 is chance; the bar for going on was 0.60.
- **The other early signals didn't help either.** Neither the "silent knowledge" signal
  nor how far the weights moved predicted switching. Both came out slightly *below*
  chance.
- **So at this scale and this early,** whether a learner will have the insight is not
  readable from its internal representation.

## Numbers (48 learners, 47.9% switched)

| Predictor (declared direction) | AUC |
|---|---|
| P_TwoNN, −Δ intrinsic dimension (primary) | 0.51 (95% CI 0.35–0.68) |
| P_corr, −Δ correlation dimension | 0.43 |
| P_rank, −Δ effective rank | 0.43 |
| K_silent, +Δ p(correct answer at step 3) | 0.37 |
| K_move, weight distance moved | 0.39 |

- **Validity.** Competence was fine: step-rule accuracy 1.0. Headroom was fine: 48%
  switched.
- **C1 and C2** both fail on development data. The futility rule then applies, because
  no compression predictor reached 0.60.

## Caveats

- **Sample size.** With 48 learners the CIs are wide (about ±0.17). A modest real
  effect, say AUC 0.6, cannot be ruled out.
- **Scope.** The test covers only *early* prediction, in one small recurrent learner.
- **Lead time.** Not evaluated, because the test phase was not run.
