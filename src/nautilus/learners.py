"""Learners.

Every learner exposes the four operations that offline phases are built from
(``nautilus.phases``): a gradient step on replayed experiences, a parameter
perturbation, a shrink of a parameter group, and an analytic FLOP count per operation.

- ``GatedNet``: the one-layer L1-gated network of Löwe et al. (2024), vectorised over
  N independent networks (numpy). ``train_lowe`` is its native online training rule,
  reproduced from the authors' code, including its quirks.
- ``NRTLearner``: a small GRU for the Number Reduction Task (torch), with an L1-gated
  readout for the final answer that it can give at any step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import erf
from torch import nn

from .tasks import DIGITS, EARLY_STEP, NRT_LEN, SSSTTrials, digit_index

GROUPS = ("weights", "gates", "all")


class PhaseLearner(ABC):
    """The interface used by awake and sleep phases."""

    @abstractmethod
    def grad_step(self, batch, lr: float, rng: np.random.Generator) -> None:
        """One SGD step on the data loss of ``batch`` (no regulariser, no noise)."""

    @abstractmethod
    def perturb(self, sigma: float, group: str, rng: np.random.Generator) -> None:
        """Add N(0, sigma^2) noise to every parameter in ``group``."""

    @abstractmethod
    def shrink(self, kind: str, amount: float, group: str) -> None:
        """Scale ``group`` down: ``l1`` soft-thresholds by ``amount``; ``l2`` multiplies
        by ``1 - amount``."""

    @abstractmethod
    def flops(self, op: str, batch_size: int = 1) -> int:
        """Analytic FLOPs of one operation: ``grad`` (forward+backward on a batch),
        ``perturb`` or ``shrink`` (whole model, worst case)."""


def soft_threshold(x, tau):
    """Proximal operator of tau*|x| (works for numpy arrays and torch tensors)."""
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.clamp(x.abs() - tau, min=0.0)
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


# ---------------------------------------------------------------------------
# Rung 1: the gated network of Löwe et al. (2024)
# ---------------------------------------------------------------------------

W_M, G_M, W_C, G_C = 0, 1, 2, 3  # column order of theta, as in the authors' code
GROUP_COLS = {"weights": [W_M, W_C], "gates": [G_M, G_C], "all": [W_M, G_M, W_C, G_C]}


@dataclass(frozen=True)
class GatedConfig:
    """Native learning rule. Defaults follow the authors' code."""

    lam: float = 0.07
    alpha: float = 0.6
    sigma_xi: float = 0.05  # SD of update noise xi
    sigma_eta: float = 0.3  # SD of output noise eta (not stated in the paper)
    reg: str = "l1"  # "l1" | "l2" | "none"
    noise_times_alpha: bool = True  # code: theta += alpha*xi; paper: theta += xi
    reg_times_alpha: bool = False  # code: g -= lam*sign(g); paper: g -= alpha*lam*sign(g)
    clip_gates: bool = True  # code: a gate that changes sign is set to 0, and a gate at 0
    # moves only if its step exceeds alpha*lam
    init: float = 0.01  # all four parameters start here
    noise_mask: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # per (wm,gm,wc,gc)
    freeze_gates: bool = False  # gates held fixed (the authors' "instructed" block)

    @classmethod
    def paper(cls, **kw) -> GatedConfig:
        """The update equations (6)-(9) exactly as printed in the paper."""
        base = dict(noise_times_alpha=False, reg_times_alpha=True, clip_gates=False)
        base.update(kw)
        return cls(**base)


def analytic_accuracy(theta, mu_m, mu_c, sigma_m, sigma_c, sigma_eta):
    """Probability of a correct choice (paper Eq. 11, as implemented in the code).

    ``theta`` has shape (..., 4); ``mu_m`` and ``mu_c`` broadcast against ``theta[..., 0]``.
    """
    em = theta[..., W_M] * theta[..., G_M]
    ec = theta[..., W_C] * theta[..., G_C]
    var = sigma_eta**2 + (em * sigma_m) ** 2 + (ec * sigma_c) ** 2
    return 0.5 * (1.0 + erf((em * mu_m + ec * mu_c) / np.sqrt(2.0 * var)))


def data_grad(theta, xm, xc, y, eta):
    """Gradient of 0.5*(g_m w_m x_m + g_c w_c x_c + eta - y)^2; inputs broadcast as (N, B)."""
    wm, gm, wc, gc = (theta[:, k : k + 1] for k in range(4))
    err = gm * wm * xm + gc * wc * xc + eta - y
    return np.stack([gm * xm * err, wm * xm * err, gc * xc * err, wc * xc * err], axis=-1)


def lowe_step(theta, xm, xc, y, eta, xi, cfg: GatedConfig):
    """One native online update for N networks. Returns (new_theta, data_gradient)."""
    grad = data_grad(theta, xm[:, None], xc[:, None], y[:, None], eta[:, None])[:, 0]
    new = theta - cfg.alpha * grad
    lam = cfg.lam * (cfg.alpha if cfg.reg_times_alpha else 1.0)
    gates = theta[:, [G_M, G_C]]
    if cfg.reg == "l1":
        new[:, [G_M, G_C]] -= lam * np.sign(gates)
    elif cfg.reg == "l2":
        new[:, [G_M, G_C]] -= lam * gates
    elif cfg.reg != "none":
        raise ValueError(cfg.reg)
    new += (cfg.alpha if cfg.noise_times_alpha else 1.0) * xi * np.asarray(cfg.noise_mask)
    if cfg.clip_gates:
        for j in (G_M, G_C):
            old, nv = theta[:, j], new[:, j]
            held = np.abs(nv) - np.minimum(cfg.alpha * cfg.lam, np.abs(nv))
            nv = np.where(old == 0, np.sign(nv) * held, nv)
            new[:, j] = np.where(old * nv < 0, 0.0, nv)
    if cfg.freeze_gates:
        new[:, [G_M, G_C]] = theta[:, [G_M, G_C]]
    return new, grad


def train_lowe(theta0, trials: SSSTTrials, cfg: GatedConfig, rng, t0: int = 0, t1=None):
    """Native online training on trials ``t0..t1-1``.

    Returns ``theta`` history of shape (N, n+1, 4), where entry ``k`` holds the
    parameters *before* trial ``t0+k``, and the data gradient of each trial (N, n, 4).
    """
    t1 = trials.y.shape[1] if t1 is None else t1
    n_nets, n = theta0.shape[0], t1 - t0
    eta = rng.normal(0.0, cfg.sigma_eta, size=(n_nets, n))
    xi = rng.normal(0.0, cfg.sigma_xi, size=(n_nets, n, 4))
    hist = np.empty((n_nets, n + 1, 4))
    grads = np.empty((n_nets, n, 4))
    hist[:, 0] = theta0
    theta = theta0.copy()
    for k in range(n):
        t = t0 + k
        theta, grads[:, k] = lowe_step(
            theta, trials.xm[:, t], trials.xc[:, t], trials.y[:, t], eta[:, k], xi[:, k], cfg
        )
        hist[:, k + 1] = theta
    return hist, grads


@dataclass
class GatedBuffer:
    """Stored experiences for N gated networks: arrays of shape (N, M)."""

    xm: np.ndarray
    xc: np.ndarray
    y: np.ndarray

    @classmethod
    def from_trials(cls, trials: SSSTTrials, t0: int, t1: int) -> GatedBuffer:
        sl = slice(t0, t1)
        return cls(trials.xm[:, sl].copy(), trials.xc[:, sl].copy(), trials.y[:, sl].copy())

    def __len__(self) -> int:
        return self.y.shape[1]

    def sample(self, rng: np.random.Generator, batch_size: int):
        idx = rng.integers(0, len(self), size=(self.y.shape[0], batch_size))
        take = lambda a: np.take_along_axis(a, idx, axis=1)  # noqa: E731
        return take(self.xm), take(self.xc), take(self.y)


class GatedNet(PhaseLearner):
    """N independent gated networks, parameters ``theta`` of shape (N, 4)."""

    N_PARAMS = 4

    def __init__(self, n: int, cfg: GatedConfig | None = None, theta=None):
        cfg = GatedConfig() if cfg is None else cfg
        self.cfg = cfg
        self.theta = np.full((n, 4), cfg.init) if theta is None else np.array(theta, float)

    def copy(self) -> GatedNet:
        return GatedNet(self.theta.shape[0], self.cfg, self.theta.copy())

    def grad_step(self, batch, lr, rng):
        xm, xc, y = batch
        eta = rng.normal(0.0, self.cfg.sigma_eta, size=y.shape)
        self.theta -= lr * data_grad(self.theta, xm, xc, y, eta).mean(axis=1)

    def perturb(self, sigma, group, rng):
        cols = GROUP_COLS[group]
        self.theta[:, cols] += rng.normal(0.0, sigma, size=(self.theta.shape[0], len(cols)))

    def shrink(self, kind, amount, group):
        cols = GROUP_COLS[group]
        if kind == "l1":
            self.theta[:, cols] = soft_threshold(self.theta[:, cols], amount)
        elif kind == "l2":
            self.theta[:, cols] *= 1.0 - amount
        else:
            raise ValueError(kind)

    def flops(self, op, batch_size=1):
        # Per network. Forward: 4 mul + 3 add (+ noise add) ~ 8; backward: 4 grads of
        # 2 mul each plus the error term ~ 10; mean and update: 2 per parameter.
        if op == "grad":
            return batch_size * (8 + 10) + 2 * self.N_PARAMS
        if op in ("perturb", "shrink"):
            return 2 * self.N_PARAMS
        raise ValueError(op)


# ---------------------------------------------------------------------------
# Rung 2: a small recurrent learner for the Number Reduction Task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NRTConfig:
    """Working defaults from a two-seed S1 smoke test; these are development knobs."""

    hidden: int = 32
    lr: float = 0.1
    gate_l1: float = 1e-2  # L1 strength on readout gates (applied as a proximal step)
    grad_noise: float = 0.0  # SD of Gaussian noise added to every parameter per update
    final_weight: float = 1.0  # weight of the final-answer loss relative to step responses
    gate_init: float = 1.0
    batch: int = 8  # strings per online update
    threshold: float = 0.9  # confidence at which the learner commits to a final answer


class NRTNet(nn.Module):
    """GRU over (digit, position) inputs. Two heads read the hidden state at every step:
    the current chained response, and the final answer through L1-regularised gates."""

    def __init__(self, hidden: int = 32, gate_init: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.GRU(len(DIGITS) + NRT_LEN, hidden, batch_first=True)
        self.resp = nn.Linear(hidden, len(DIGITS))
        self.final = nn.Linear(hidden, len(DIGITS))
        self.gate = nn.Parameter(torch.full((hidden,), float(gate_init)))

    @staticmethod
    def encode(digit_idx: torch.Tensor) -> torch.Tensor:
        b = digit_idx.shape[0]
        d = nn.functional.one_hot(digit_idx, len(DIGITS)).float()
        pos = torch.eye(NRT_LEN).expand(b, NRT_LEN, NRT_LEN)
        return torch.cat([d, pos], dim=-1)

    def forward(self, digit_idx: torch.Tensor):
        h, _ = self.rnn(self.encode(digit_idx))
        return h, self.resp(h), self.final(h * self.gate)


def nrt_loss(net: NRTNet, digit_idx, resp_idx, final_weight: float):
    """Cross-entropy on responses r1..r7 (steps 2..8) plus the final answer at every step."""
    h, resp_logits, final_logits = net(digit_idx)
    ce = nn.functional.cross_entropy
    loss_resp = ce(resp_logits[:, 1:].reshape(-1, 3), resp_idx.reshape(-1))
    answer = resp_idx[:, -1:].expand(-1, NRT_LEN)
    loss_final = ce(final_logits.reshape(-1, 3), answer.reshape(-1))
    return loss_resp + final_weight * loss_final, (h, resp_logits, final_logits)


def answer_readout(final_logits: torch.Tensor, answer_idx: torch.Tensor, threshold: float):
    """When does the learner answer, and is it right?

    The learner answers at the first step whose final-answer confidence reaches
    ``threshold``, or at step 8. Returns (answer_step 1..8, correct, early), where
    ``early`` means answered at step <= EARLY_STEP.
    """
    p = torch.softmax(final_logits, dim=-1)
    conf, choice = p.max(dim=-1)
    reached = conf >= threshold
    reached[:, -1] = True
    step = reached.float().argmax(dim=1)  # first True
    chosen = choice.gather(1, step[:, None])[:, 0]
    correct = chosen == answer_idx
    answer_step = step + 1
    return answer_step, correct, answer_step <= EARLY_STEP


@dataclass
class NRTBuffer:
    """Stored NRT experiences as class indices: digits (M, 8), responses (M, 7)."""

    digits: torch.Tensor
    resps: torch.Tensor

    @classmethod
    def from_values(cls, digits: np.ndarray, resps: np.ndarray) -> NRTBuffer:
        return cls(torch.as_tensor(digit_index(digits)), torch.as_tensor(digit_index(resps)))

    def __len__(self) -> int:
        return self.digits.shape[0]

    def sample(self, rng: np.random.Generator, batch_size: int):
        idx = torch.as_tensor(rng.integers(0, len(self), size=batch_size))
        return self.digits[idx], self.resps[idx]


class NRTLearner(PhaseLearner):
    def __init__(self, cfg: NRTConfig | None = None, seed: int = 0):
        cfg = NRTConfig() if cfg is None else cfg
        self.cfg = cfg
        torch.manual_seed(seed)
        self.net = NRTNet(cfg.hidden, cfg.gate_init)

    def _groups(self, group: str) -> list[nn.Parameter]:
        if group == "gates":
            return [self.net.gate]
        if group == "weights":
            return [p for n, p in self.net.named_parameters() if n != "gate"]
        if group == "all":
            return list(self.net.parameters())
        raise ValueError(group)

    def _sgd(self, loss, lr):
        self.net.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                p -= lr * p.grad

    def grad_step(self, batch, lr, rng):
        loss, _ = nrt_loss(self.net, *batch, self.cfg.final_weight)
        self._sgd(loss, lr)

    def perturb(self, sigma, group, rng):
        g = torch.Generator().manual_seed(int(rng.integers(2**62)))
        with torch.no_grad():
            for p in self._groups(group):
                p += sigma * torch.randn(p.shape, generator=g)

    def shrink(self, kind, amount, group):
        with torch.no_grad():
            for p in self._groups(group):
                if kind == "l1":
                    p.copy_(soft_threshold(p, amount))
                elif kind == "l2":
                    p.mul_(1.0 - amount)
                else:
                    raise ValueError(kind)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def forward_flops(self, batch_size: int = 1) -> int:
        h, i, o = self.cfg.hidden, len(DIGITS) + NRT_LEN, len(DIGITS)
        gru = 3 * (2 * i * h + 2 * h * h) + 12 * h  # 3 gate matmuls + elementwise
        heads = 2 * (2 * h * o) + h  # two linear heads + gating
        return batch_size * NRT_LEN * (gru + heads)

    def flops(self, op, batch_size=1):
        if op == "grad":  # backward ~ 2x forward, plus the SGD update
            return 3 * self.forward_flops(batch_size) + 2 * self.n_params()
        if op in ("perturb", "shrink"):
            return 2 * self.n_params()
        raise ValueError(op)

    def online_trial(self, digit_idx, resp_idx, rng: np.random.Generator):
        """Native training on one batch of trials: respond, then learn.

        Returns (answer_step, correct, early) measured *before* the update, as in a
        behavioural experiment.
        """
        cfg = self.cfg
        loss, (_, _, final_logits) = nrt_loss(self.net, digit_idx, resp_idx, cfg.final_weight)
        with torch.no_grad():
            out = answer_readout(final_logits, resp_idx[:, -1], cfg.threshold)
        self._sgd(loss, cfg.lr)
        if cfg.gate_l1 > 0:
            self.shrink("l1", cfg.lr * cfg.gate_l1, "gates")
        if cfg.grad_noise > 0:
            self.perturb(cfg.grad_noise, "all", rng)
        return out

    @torch.no_grad()
    def hidden_states(self, digit_idx: torch.Tensor) -> torch.Tensor:
        """Hidden states (M, 8, H) for a probe set."""
        return self.net(digit_idx)[0]
