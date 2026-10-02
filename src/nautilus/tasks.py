"""Tasks.

Rung 1: the symbolic Spontaneous Strategy Switch Task (SSST) of Löwe et al. (2024),
arXiv 2302.11351. Two scalar inputs per trial, motion ``x_m`` and colour ``x_c``.
Colour becomes predictive of the answer without warning.

Rung 2: the Number Reduction Task (NRT) of Wagner et al. (2004). Strings of 8 digits
from {1, 4, 9} are reduced left to right; a hidden mirror structure makes the 2nd
response equal to the final answer.

Where the paper text and the authors' code (gitlab.com/aloewe/insightnets) disagree,
``SSSTConfig()`` follows the code, which produced the published numbers, and
``SSSTConfig.paper()`` follows the text. See ``reports/rung1_replication.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FITTED_MEANS_CSV = REPO_ROOT / "configs" / "lowe2024_fitted_motion_means.csv"

# ---------------------------------------------------------------------------
# Rung 1: Spontaneous Strategy Switch Task
# ---------------------------------------------------------------------------

N_COH = 5  # motion coherence levels 5, 10, 20, 30, 45 %; index 0 is the hardest
COH_PERCENT = (5, 10, 20, 30, 45)
COH_PER_100 = (30, 10, 20, 20, 20)  # trials per 100 at each coherence level
PRETRAIN_COHS = (2, 3, 4)  # pre-training uses the three easiest levels

PHASE_PRETRAIN, PHASE_MOTION, PHASE_COLOUR = 0, 1, 2


@dataclass(frozen=True)
class SSSTConfig:
    """Curriculum and input statistics. Defaults follow the authors' code."""

    n_pretrain: int = 800  # code: 8 blocks; paper text: 6 blocks
    n_motion: int = 200  # 2 blocks, colour not predictive
    n_colour: int = 400  # 4 blocks, colour predictive (unannounced)
    block: int = 100
    mu_c: float = 0.22  # colour input mean once predictive
    sigma_m: float = 0.1  # motion input SD (code; paper text once says 0.01)
    sigma_c: float = 0.01  # colour input SD
    pm1_targets: bool = False  # code: y in {0, 1}; paper equations: y in {-1, +1}
    random_colour_sign: bool = False  # paper: uncorrelated colour is +-mu_c at random;
    # code: uncorrelated colour has mean 0
    control: bool = False  # colour never becomes predictive

    @classmethod
    def paper(cls, **kw) -> SSSTConfig:
        """The task as described in the paper's text rather than the code."""
        base = dict(n_pretrain=600, pm1_targets=True, random_colour_sign=True)
        base.update(kw)
        return cls(**base)

    def as_control(self) -> SSSTConfig:
        return replace(self, control=True)

    @property
    def n_trials(self) -> int:
        return self.n_pretrain + self.n_motion + self.n_colour

    @property
    def colour_onset(self) -> int:
        """Index of the first trial on which colour can be predictive."""
        return self.n_pretrain + self.n_motion


@dataclass
class SSSTTrials:
    """Trial sequence for N networks: arrays of shape (N, T) unless noted."""

    coh: np.ndarray  # int coherence index 0..4
    y: np.ndarray  # target, in {0,1} or {-1,+1}
    xm: np.ndarray  # motion input
    xc: np.ndarray  # colour input
    mu_m: np.ndarray  # motion mean of the trial's coherence level (unsigned)
    mu_c: np.ndarray  # colour mean if colour is predictive on this trial, else 0
    phase: np.ndarray  # (T,) phase code


def load_fitted_motion_means(path: Path = FITTED_MEANS_CSV) -> np.ndarray:
    """The authors' 99 per-network motion means, shape (99, 5)."""
    return np.loadtxt(path, delimiter=",", comments="#")


def _balanced_block(rng: np.random.Generator, block: int) -> tuple[np.ndarray, np.ndarray]:
    """One block with exact coherence counts, each split evenly between the two answers."""
    scale = block // 100
    coh, lab = [], []
    for k, c in enumerate(COH_PER_100):
        n = c * scale
        coh += [k] * n
        lab += [0] * (n // 2) + [1] * (n - n // 2)
    order = rng.permutation(len(coh))
    return np.asarray(coh)[order], np.asarray(lab)[order]


def make_ssst_trials(
    cfg: SSSTConfig, mu_m_table: np.ndarray, rng: np.random.Generator
) -> SSSTTrials:
    """Generate trials for ``len(mu_m_table)`` networks (one row of means per network)."""
    mu_m_table = np.atleast_2d(np.asarray(mu_m_table, dtype=float))
    n, t = len(mu_m_table), cfg.n_trials
    coh = np.empty((n, t), dtype=int)
    lab = np.empty((n, t), dtype=int)
    tp = cfg.n_pretrain
    coh[:, :tp] = rng.choice(PRETRAIN_COHS, size=(n, tp))
    lab[:, :tp] = rng.integers(0, 2, size=(n, tp))
    n_blocks = (cfg.n_motion + cfg.n_colour) // cfg.block
    for i in range(n):
        parts = [_balanced_block(rng, cfg.block) for _ in range(n_blocks)]
        coh[i, tp:] = np.concatenate([p[0] for p in parts])
        lab[i, tp:] = np.concatenate([p[1] for p in parts])

    phase = np.full(t, PHASE_PRETRAIN)
    phase[tp : cfg.colour_onset] = PHASE_MOTION
    phase[cfg.colour_onset :] = PHASE_COLOUR

    sign = 2 * lab - 1  # +-1 version of the answer
    y = sign.astype(float) if cfg.pm1_targets else lab.astype(float)
    mu_m = np.take_along_axis(mu_m_table, coh, axis=1)
    xm = rng.normal(y * mu_m, cfg.sigma_m)

    predictive = (phase == PHASE_COLOUR) & (not cfg.control)
    mu_c = np.where(predictive[None, :], cfg.mu_c, 0.0) * np.ones((n, 1))
    if cfg.random_colour_sign:
        rand_sign = rng.choice([-1.0, 1.0], size=(n, t))
        colour_mean = np.where(predictive[None, :], y * cfg.mu_c, rand_sign * cfg.mu_c)
    else:
        colour_mean = y * mu_c
    xc = rng.normal(colour_mean, cfg.sigma_c)
    return SSSTTrials(coh=coh, y=y, xm=xm, xc=xc, mu_m=mu_m, mu_c=mu_c, phase=phase)


# ---------------------------------------------------------------------------
# Rung 2: Number Reduction Task
# ---------------------------------------------------------------------------

DIGITS = (1, 4, 9)
NRT_LEN = 8  # digits per string; 7 responses
EARLY_STEP = 3  # the 2nd response (= final answer under the mirror) is known after digit 3


def nrt_rule(a: int, b: int) -> int:
    """'Same' rule: two equal digits give that digit. 'Different': the remaining digit."""
    if a == b:
        return a
    (rest,) = set(DIGITS) - {a, b}
    return rest


def nrt_responses(digits) -> list[int]:
    """Chained responses r1..r7 for an 8-digit string."""
    r = [nrt_rule(digits[0], digits[1])]
    for d in digits[2:]:
        r.append(nrt_rule(r[-1], d))
    return r


def _digit_for(prev: int, resp: int) -> int:
    """The unique digit d with nrt_rule(prev, d) == resp."""
    return prev if resp == prev else nrt_rule(prev, resp)


def make_nrt_strings(
    n: int, rng: np.random.Generator, mirror: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``digits`` (n, 8) and ``responses`` (n, 7) as digit values.

    ``mirror=True``: responses r2..r7 follow the pattern A B C C B A (Wagner et al. 2004),
    so r2 equals the final answer r7. ``mirror=False``: digits are i.i.d. uniform.
    """
    digits = np.empty((n, NRT_LEN), dtype=int)
    resps = np.empty((n, NRT_LEN - 1), dtype=int)
    for i in range(n):
        if mirror:
            d1, d2 = rng.choice(DIGITS, size=2)
            r1 = nrt_rule(d1, d2)
            a, b, c = rng.choice(DIGITS, size=3)
            r = [r1, a, b, c, c, b, a]
            d = [d1, d2] + [_digit_for(r[k - 1], r[k]) for k in range(1, 7)]
        else:
            d = list(rng.choice(DIGITS, size=NRT_LEN))
            r = nrt_responses(d)
        digits[i], resps[i] = d, r
    return digits, resps


def digit_index(values: np.ndarray) -> np.ndarray:
    """Map digit values {1,4,9} to class indices {0,1,2}."""
    lut = np.full(10, -1)
    for k, v in enumerate(DIGITS):
        lut[v] = k
    return lut[np.asarray(values)]
