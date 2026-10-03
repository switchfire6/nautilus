# P1 results: sleep versus equal awake compute (H1)

*Protocol: `docs/protocols/P1_sleep_vs_awake.md`, locked 2026-10-02 (lock commit
`0b1728d`, hashes in `docs/protocols/P1_lock.json`). Test seeds were run only after
the lock. Raw results are in `runs/p1/test/`. The summary is `reports/P1_results.json`
(SHA-256 `0127533a…4d8c03`), produced by the locked `scripts/p1_analyse.py`.*

## 1. Plain-language summary

**Sleep did not beat awake practice.** After a first learning session, we gave the
learners extra offline compute in one of two forms. In both cases they used only the
experiences they had already stored.

- **Awake practice:** going over those experiences again.
- **Sleep:** the same replay, broken up by short "dream" bursts of random noise, with
  optional scaling-down of connections.

Both forms used the same amount of arithmetic, and each was tuned on separate
development seeds with the same budget of 32 settings. Both helped a lot compared
with doing nothing, but **equally**:

| | Nothing (baseline) | Awake practice | Sleep | Told the shortcut (ceiling) |
|---|---|---|---|---|
| Rung 1 (Löwe's perceptual task), 1,980 networks | 48% | **66%** | **64%** | 99% |
| Rung 2 (number puzzle), 96 learners | 20% | **76%** | **76%** | 100% |

*(Share of learners that discovered the hidden shortcut.)*

**What actually helped was replay: going over stored experiences.**

- **Noise alone did nothing.** The noise-only arm stayed at baseline (49%, 22%).
- **Scaling down never helped on development seeds.** Tuning chose *no*
  scaling-down for sleep on both rungs, so the shrink-only arm was a no-op.
- **More replay helped more.** The replay-only arm had more replay steps than
  either awake or sleep (it is not compute-matched). It did best of all the offline
  arms: 68% and 85%.

In these models, "sleeping on it" works because it is **extra practice**, not because
sleep adds anything special. The pre-registered prediction (H1) fails on both rungs.
This is the calibrated negative the protocol anticipated (§8), and the same pattern
as both predecessor projects.

## 2. Pre-declared criteria

| Criterion | Result | Passed? |
|---|---|---|
| **C1** rung 1: sleep − awake share, 95% CI lower bound > 0 **and** sleep higher in ≥ 15/20 batches | −1.4 points, CI [−3.0, +0.05]; sleep higher in 6/20 batches | **No** |
| **C2** rung 2: sleep − awake share, 95% CI lower bound > 0 **and** sleep higher in ≥ 3/4 blocks | 0.0 points, CI [−3.1, +3.1]; sleep higher in 1/4 blocks | **No** |

## 3. Validity checks (all pass, so the criteria are interpretable)

| Check | Rung 1 | Rung 2 |
|---|---|---|
| V1 headroom (baseline 10–70%) | 48.4% ✓ | 19.8% ✓ |
| V2 competence | easy-motion accuracy 0.84 > 0.75 ✓ | step-rule accuracy ≥ 0.9997 ✓ |
| V3 ceiling ≥ 90% | 99.0% ✓ | 100% ✓ |
| V4 awake/sleep FLOPs within 2% | ✓ | ✓ |
| V5 replicate pass (subset recomputed from scratch) | identical ✓ | identical ✓ |

## 4. All arms (test seeds)

| Arm | Rung 1 share | Rung 2 share | Offline steps (r1 / r2) |
|---|---|---|---|
| Baseline | 48.4% | 19.8% | 0 / 0 |
| Awake (tuned: plain replay, lr 0.6 / 0.1, no added L1 or noise) | 65.9% | 76.0% | 32 / 202 |
| Sleep (tuned: replay + REM noise 1×, no downscaling) | 64.4% | 76.0% | 36 / 252 |
| Replay only (sleep's step count, all replay; not FLOP-matched) | 68.2% | 85.4% | 36 / 252 |
| Noise only | 48.9% | 21.9% | 36 / 252 |
| Shrink only (no-op: tuned sleep has no shrink) | 48.4% | 19.8% | 36 / 252 |
| Ceiling | 99.0% | 100% | — |

- **Offline vs nothing.** Both offline arms beat baseline clearly:
  - rung 1: sleep +16.0 points (CI 14.5–17.5), awake +17.4 points (15.6–19.3);
  - rung 2: both +56 points (CI about 46–67).
- **Time to switch** (secondary). Both arms make learners switch earlier than
  baseline, by about the same amount.
  - Rung 1, first session-2 bin above threshold: baseline 1.27, awake 0.66, sleep 0.67.
  - Rung 2: baseline 4.1, awake 3.0, sleep 2.9 bins of 500 strings.
- **Löwe criterion** (secondary, rung 1). Baseline 25.5%, awake 37.8%, sleep 39.3%.
  It agrees: no meaningful sleep advantage.

## 5. H2, descriptive only (rung 2, step-3 hidden states, change over session 2)

| Arm | Group | Δ TwoNN dimension | Δ effective rank |
|---|---|---|---|
| Baseline | switchers (n = 19) | −0.30 | +0.88 |
| Baseline | non-switchers (n = 77) | +0.03 | +0.95 |
| Awake | switchers / non-switchers | −0.21 / −0.24 | +0.51 / +0.78 |
| Sleep | switchers / non-switchers | −0.21 / −0.16 | +0.50 / +0.70 |

- **Baseline.** Without an offline phase, only the learners that switch show a drop
  in intrinsic dimension, which fits "insight is compression".
- **After an offline phase,** non-switchers drop just as much, so the drop tracks
  replay training rather than insight itself.
- **Effective rank** rises in every group.
- **Verdict.** There is no clean compression signature of insight here. This is
  descriptive only; H2 is for P2.

## 6. Deviations and disclosures

- **No post-lock deviations.** The test and analysis ran exactly as locked.
- **Pre-lock amendments (protocol §10)** were made on development data, before the
  lock:
  - The most consequential is amendment 9. The v1 offline budget made every replaying
    arm saturate near 100% on development seeds. The budget was recalibrated by a rule
    that looks only at plain replay, the awake reference, never at sleep.
  - The superseded development runs are kept in `runs/p1/dev_superseded/`.
- **Within-protocol limitation.** Tuning picked "no scaling-down" for sleep, so the
  synaptic-downscaling idea was effectively tested only on development seeds. There,
  every downscaling setting was no better, and strong downscaling broke rung-2
  learners. The test seeds tell us nothing new about downscaling.

## 7. What this implies (protocol §8)

**Calibrated negative: "sleep is extra compute."** In both an L1-gated insight
network and a recurrent number-puzzle learner, an offline phase helps insight only
through replay. Neither its noise nor its scaling-down adds anything, at matched
compute, over well-tuned awake practice.

Per the brief's S2 exit rule, H1 does **not** reach GO. H2 (does internal compression
come first, and predict insight?) and H3 (self-similarity) are separate questions,
and they remain open.
