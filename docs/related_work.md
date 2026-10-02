# Related work (S0 literature check)

*Written 2026-10-02, before any experiment. This is a focused novelty check
for the Nautilus hypotheses (`docs/PROJECT_BRIEF.md` §3). It is not a full
survey. Method and limits are in §6.*

---

## 1. Bottom line (plain language)

- **Sleep helps human insight.** This is established, and recent evidence links
  it to the brain *scaling down* its connections during deeper sleep, which is
  a form of regularisation.
- **Simple neural networks can have human-like "aha" moments.** This is
  established too (Löwe et al., 2024). They do it through noise and
  regularised attention gates, with knowledge building up silently first.
- **Nobody seems to have put the two together.** We found no test of a
  sleep-like phase in such a network against an **equal amount of extra
  waking computation**. The same research group explicitly left sleep in
  networks for future work. This is our H1.
- **"Insight is compression" is an old idea with modern evidence:**
  - Schmidhuber's compression-progress theory;
  - grokking research, where a network's sudden generalisation coincides with
    its internal representation getting simpler.

  What seems untested is whether internal compression **starts before** the
  behavioural "aha", so that it can predict it, and whether sleep drives it.
  That is our narrowed H2.
- **Fractals appear across AI research** in how representations, language and
  training behave, and in fractal compression and fractal pretraining. A
  **self-similarity bias for discovering hidden rules** was not found. That is
  H3.
- **Thinking without language** (latent reasoning, mental imagery) is a
  crowded frontier-lab field. We stay out of it.

---

## 2. What already exists

### 2.1 Sleep and insight in humans

| Work | Finding | Relevance |
|---|---|---|
| Wagner et al., *Nature* 2004 ([record](https://research.uni-luebeck.de/en/publications/sleep-inspires-insight/)) | **Number Reduction Task** (NRT): more than twice as many people found the hidden shortcut after sleep as after wakefulness | Our rung-2 task |
| Lacaux et al., *Sci. Adv.* 2021 (cited in Löwe et al. 2024) | The sleep-onset (N1) period is a "creative sweet spot" for a hidden-rule task | Timing and depth of sleep matter |
| **Löwe, Petzka, Tzegka & Schuck, *PLOS Biology* 2025** ([paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003185)) | 20-minute nap; insight in **85.7% after N2 sleep**, 63.6% after N1 and 55.5% awake (n = 68). A steeper EEG spectral slope, a marker of synaptic downscaling, predicted insight. **No network simulation of sleep.** | The strongest motivation for H1: deeper sleep acts like regularisation |

### 2.2 Computational models of insight and incubation

| Work | What it shows | Gap for us |
|---|---|---|
| **Löwe, Touzo, Muhle-Karbe, Summerfield, Saxe & Schuck 2024** ([arXiv](https://arxiv.org/abs/2302.11351)) | A tiny network (two inputs, multiplicative gates, L1 penalty on the gates, gradient noise) reproduces human insight: **delay, suddenness, selectivity**. "Silent knowledge" accumulates while the gate is suppressed. Insight is measured by sigmoid fits (switch point and slope). | Their own words: experiments ran "in an uninterrupted fashion during daylight", and they **suggest regularisation during sleep** might matter. Not tested in networks (the thesis below confirms this). **Our rung 1 starts here.** |
| Löwe, PhD thesis, Hamburg 2024 ([record](https://ediss.sub.uni-hamburg.de/handle/ediss/11657)) | Chapter 3 is the network model (no sleep); chapter 4 is human naps | Confirms the gap |
| Lerner, CCN abstract ([PDF](https://www2.securecms.com/CCNeuro/docs-0/5928daeb68ed3f7a4e8a2571.pdf)) | A spiking hippocampus–prefrontal model of the **NRT**. Sleep's time-compressed replay (×80) lets Hebbian learning pick up the **mirror structure**; after sleep, prediction is fast. | The mechanism is specific to spiking timescales: backpropagation has no such window. Its control is "no sleep learning", not equal awake compute. |
| Hélie & Sun, *Psych. Review* 2010 ([record](https://cindy.informatik.uni-bremen.de/www3/www.sfbtr8.spatial-cognition.de/aigaion/index.php/publications/show/2213.html)) | Explicit–implicit interaction theory (CLARION): implicit, non-verbal processing continues during incubation; insight is when its result crosses into explicit knowledge. It matched human incubation data. | **The closest conceptual twin of the owner's "subconscious" idea.** A cognitive architecture fitted to human data, not a learning test against equal compute. |
| Kralik et al. 2016 ([record](https://dspace.kaist.ac.kr/handle/10203/216084)) | Insight by restructuring internal beliefs with evidence gathered during incubation; matched monkey data; ran on a robot | No compute-matched test |

### 2.3 Sleep-like phases in artificial systems (memory and efficiency, not insight)

- **Sleep against forgetting:**
  - sleep-like replay ([Tadros et al., 2022](https://www.nature.com/articles/s41467-022-34938-7));
  - wake–sleep consolidated learning ([2024](https://arxiv.org/pdf/2401.08623));
  - "dreaming" Hopfield networks, which unlearn spurious memories
    ([Fachechi et al., 2019](https://arxiv.org/pdf/1810.12217), after Hopfield,
    and Crick & Mitchison, 1983).
- **Abstraction:** [DreamCoder](https://arxiv.org/pdf/2006.08381) alternates
  waking problem-solving, *abstraction sleep* (compressing solutions into
  reusable concepts) and *dream sleep* (practising on imagined problems).
- **Language models:**
  - [sleep-time compute](https://letta.com/blog/sleep-time-compute) (2025):
    pre-thinking offline cuts query-time compute about 5×. This is an
    equal-compute comparison, but about efficiency;
  - a [Google/Cornell "Sleep" phase](https://hyper.ai/en/papers/2606.03979)
    (2026) of consolidation plus dreaming, for continual learning.

### 2.4 Insight as compression

- **Schmidhuber, "Driven by compression progress"** ([arXiv 2008](https://arxiv.org/abs/0812.4360)).
  Interestingness, curiosity and discovery are improvements in an agent's
  compressor. This is the classic statement of our theory. *We test it; we do
  not claim it.*
- **Grokking:**
  - [complexity rises during memorisation, then falls](https://arxiv.org/html/2412.09810v2)
    when the network finds the simpler pattern; only regularised networks do
    this;
  - the [local intrinsic dimension of activations drops](https://neurips.cc/virtual/2022/57156)
    at the moment of sudden generalisation.

  **So H2's "insight coincides with compression" is largely known.** The open
  part is whether compression *leads* behaviour and predicts insight, and
  whether sleep drives it.
- [CompressARC](https://arxiv.org/abs/2512.06104) (2025) solves about 20% of
  ARC puzzles purely by minimising description length at test time. Its
  76K-parameter model needs no pretraining.

### 2.5 Fractals and self-similarity in AI

| Work | What it does |
|---|---|
| [Fractal patterns in language](https://arxiv.org/abs/2402.01825) (NeurIPS 2024) | Language is self-similar (Hurst ≈ 0.70). Fractal parameters predict downstream performance better than perplexity alone. |
| [Correlation (fractal) dimension of language model representations](https://arxiv.org/pdf/2405.06321); [local intrinsic dimension](https://arxiv.org/pdf/2506.01034) | Track training phases, flag hallucination; representations lie on low-dimensional, stratified manifolds |
| [Trainability boundary is fractal](https://arxiv.org/pdf/2402.06184) (Sohl-Dickstein 2024) | The fractal boundary between hyperparameters that train and those that diverge |
| [Neural Collages](https://arxiv.org/pdf/2204.07673) (NeurIPS 2022) | Differentiable fractal (self-similarity) codes; fast fractal compression of images |
| [FractalDB](https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf) (2020) | Pretraining on fractal images alone rivals natural images |
| [Fractal generative models](https://arxiv.org/abs/2502.17437) (Kaiming He's group, 2025); [tiny recursive reasoners](https://arxiv.org/abs/2510.04871) (2025) | Self-similar or recursive architectures |

**Not found:** a self-similarity or fractal compression *prior* used to
discover hidden rules, tested against a matched non-fractal control.

### 2.6 Thinking without language (context only; we avoid this field)

Examples of the field:

- [Coconut](https://arxiv.org/abs/2412.06769), with a
  [survey of latent reasoning](https://arxiv.org/pdf/2507.06203);
- latent "mental imagery" tokens
  ([1](https://arxiv.org/pdf/2506.17218), [2](https://arxiv.org/html/2510.24514v1));
- [Latent Program Networks](https://arxiv.org/abs/2411.08706).

It is crowded and needs frontier-lab compute.

### 2.7 Context: AI at frontier scale (2026)

On 8 September 2026, OpenAI reported that about 10,000 agents proved
finite-time blow-up for Navier–Stokes with smooth forcing (Clay statements
C and D), checked in Lean.

- **Cost:** Sébastien Bubeck estimated it at "several million dollars".
- **What's still open:** the unforced question remains open.
- **Credit dispute:** a group including an Anthropic researcher posted related
  work hours earlier.

Sources: [Quanta](https://www.quantamagazine.org/ai-has-solved-one-of-maths-1-million-millennium-prize-problems-20260908/),
[ScienceABC](https://www.scienceabc.com/pure-sciences/navier-stokes-openai-singularity-what-was-proved-what-is-open),
[Axios](https://www.axios.com/2026/09/08/openai-math-solution-navier-stokes-credit).

**Lesson:** generating ideas was cheap at that scale, and verification made
them count. Small projects should ask mechanism questions, not race on
capability.

---

## 3. What remains open (our questions)

1. **H1.** Does a sleep-like offline phase (replay, noise, scaling down) in an
   insight-capable network cause more or earlier insight than an **equal
   amount of tuned awake training**?
   - The motivation: the human N2 and spectral-slope findings, and the network
     model's dependence on regularisation and noise.
   - Not found in the literature.
2. **H2, narrowed.** Does internal compression **precede** the behavioural
   switch, giving a measurable early marker of silent knowledge, and does sleep
   accelerate it? (That compression *coincides* with sudden generalisation is
   known from grokking.)
3. **H3.** Does a self-similarity bias during sleep help discover self-similar
   rules specifically, against matched rules that are not self-similar?

---

## 4. Implications for S1 and the first protocol

- **Replicate before extending.** Reimplement Löwe et al.'s gated network and
  task. Reproduce their headline statistics (insight in only some networks,
  with delay and suddenness) before adding sleep. No public code link was found
  in the paper text, so we reimplement from the equations:
  L = ½(g_m·w_m·x_m + g_c·w_c·x_c + η − y)² + λ(|g_m| + |g_c|), plus
  gradient noise.
- **The awake arm must be strong.** It gets its own tuned L1 strength and
  noise. Löwe et al. found that these two ingredients *produce* insight while
  awake, so sleep must beat their best continuous use.
- **Use their insight measures** (sigmoid switch point and slope, share of
  networks switching), so results are comparable.
- **Rung 2** reuses Wagner's NRT structure, as also modelled by Lerner.
  Insight is responding after the 2nd step.
- **Pre-declare one primary compression measure** for H2, because these
  measures are noisy in small networks.

---

## 5. Biological background beyond the above

Synaptic downscaling during sleep (Tononi & Cirelli, 2014), and the
"overfitted brain" theory that dreams prevent overfitting (Hoel, 2021). Both
are cited in Löwe et al. (2024) and were not re-searched here.

---

## 6. Method and limits

- **Searches.** About 25 web searches on 2026-10-02 (a few in extended mode),
  plus close reads of:
  - Löwe et al. (2024), full text;
  - Lerner's CCN abstract, full text;
  - the Löwe (2025) PLOS Biology paper, via a targeted summary;
  - the Löwe thesis record.
- **Limits.** Absence in these searches is not proof of absence. Re-check
  before any write-up, especially:
  - later work citing Löwe et al. (2024, 2025);
  - work on "sleep" and "grokking";
  - computational models of the Number Reduction Task.
- **Not read in full:** Hélie & Sun (2010) and Kralik et al. (2016). They are
  described from abstracts and from Löwe et al.'s summary.
