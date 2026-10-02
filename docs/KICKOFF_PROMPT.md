# Kickoff prompt for the first Nautilus session

Start Claude Code (or the T3 Code harness) with the working folder set to
`/home/switchfire/nautilus`, then paste everything below the line.

---

You're continuing my research project **Nautilus** in this folder. Read
`CLAUDE.md`, `docs/PROJECT_BRIEF.md` and `docs/related_work.md` fully before
doing anything else. They define the question, the hypotheses (H1–H3), the
stages and the binding research rules.

**Where things stand (2026-10-02):**

- **S0 (literature check) is done.** The gap is confirmed: nobody has tested
  a sleep-like offline phase in an insight-capable network against equal
  awake compute.
- **Decided:** no "quiet rest" arm. An idle AI does not change, so rest equals
  the free baseline snapshot.
- **Nothing is committed yet. There is no GitHub repo.** Ask me whether to
  make an initial commit and create the repo.

**Do S1 (engineering), in order, without intermediate check-ins:**

1. **Environment.** Create `.venv`. Install CPU PyTorch (same pins as
   `/home/switchfire/planaria/requirements.txt`), numpy, scipy, pytest and
   ruff. Record the versions in `requirements.txt`. Add `pyproject.toml` with a
   `src/nautilus` package.
2. **Rung 1: replicate Löwe et al. (2024)**, arXiv 2302.11351, Methods
   section. Reimplement their one-layer L1-gated network and the symbolic
   Spontaneous Strategy Switch Task:
   - **Model:** two scalar inputs (motion x_m, colour x_c); weights w;
     multiplicative gates g. Loss
     L = ½(g_m·w_m·x_m + g_c·w_c·x_c + η − y)² + λ(|g_m| + |g_c|).
   - **Training:** updates after every trial, with Gaussian noise added to
     the weight and gate updates. Parameters: **λ = 0.07, α = 0.6,
     σ_ξ = 0.05**.
   - **Curriculum (blocks of 100 trials):**
     - 6 pre-training blocks;
     - 2 blocks of "motion phase" (colour random);
     - 5 blocks of "motion + colour phase" (colour predictive, unannounced).
     - Motion coherence has 5 levels; take the input values from their
       Methods.
   - **Insight classification:** fit a sigmoid to accuracy at the lowest
     coherence level. A network counts as insight if its slope at the
     inflection point exceeds the maximum slope of 99 *control* networks,
     which are trained on the same task but with colour never predictive.
   - **Targets to reproduce:**
     - **46/99 (46.5%)** of networks classed as insight;
     - mean switch delay **≈ 175 ± 105 trials** after colour onset;
     - insight frequency rising with σ_ξ up to about 0.05 and falling with λ
       (their Fig. 4);
     - "silent knowledge": larger colour-gate gradients in the first trials
       for insight networks.

     For comparison, their hidden-layer variant gave 18/99 at λ = 0.002 and
     α = 0.1, and humans gave 49/99.
3. **Rung 2: Number Reduction Task** (Wagner et al., 2004). Digit strings
   from {1, 4, 9}:
   - the "same" rule: two identical digits give that digit;
   - the "different" rule: two different digits give the remaining one;
   - responses are chained left to right;
   - **hidden structure:** the response sequence mirrors, so the 2nd response
     equals the final answer.

   Build a small learner whose insight is answering early with accuracy
   maintained.
4. **Phases.**
   - **Awake:** extra training on the same stored experiences, with its own
     tunable regularisation and noise.
   - **Sleep:** offline, with no new data: replay, noise, scaling-down
     (L1/L2 shrink).
   - **Ablations:** replay only, noise only, shrink only.
   - **Compute accounting:** update steps and FLOPs. Verify that the budgets
     are equal.
5. **Measures:**
   - insight: sigmoid switch point and slope; share of learners switching;
     delay;
   - compression: TwoNN intrinsic dimension, correlation dimension, effective
     rank, gate sparsity;
   - lead time: the compression change point minus the behavioural switch
     point.
6. **Tests** for every task, phase and measure. Keep `pytest` and
   `ruff check` passing.

✋ **Stop at the end of S1.** Show me, in plain language first:

- the rung-1 replication numbers against the published ones;
- the timings;
- anything surprising.

Then we write the first protocol (S2). Keep token use lean: small direct
scripts, no large multi-agent pipelines. Don't commit or push unless I ask.
