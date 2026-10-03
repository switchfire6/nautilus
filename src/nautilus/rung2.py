"""Rung 2: train the NRT learner online and record behaviour and compression.

The curriculum mirrors rung 1:

1. *Instruction*: the step rules are taught with Adam on strings without structure,
   standing in for the instructions and practice human participants get. This is
   needed because the NRT rule is modular addition (r = -(a+b) mod 3 with digits
   coded 0,1,2), on which plain SGD stalls for a long time.
2. *Pre-phase*: online SGD practice on strings without structure.
3. *Main phase*: (unannounced) every string has the mirror structure, so the final
   answer is already known after digit 3. Control learners never see the mirror.

Insight = answering at step <= 3 while staying correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .learners import NRTConfig, NRTLearner, nrt_loss
from .measures import correlation_dimension, effective_rank, gate_sparsity, twonn_dimension
from .tasks import EARLY_STEP, digit_index, make_nrt_strings


@dataclass(frozen=True)
class NRTCurriculum:
    n_instruct: int = 600  # Adam updates teaching the step rules
    instruct_lr: float = 1e-2
    instruct_batch: int = 32
    n_pre: int = 4000  # strings without structure
    n_main: int = 4000  # strings with the mirror (unless control)
    probe_every: int = 200  # trials between compression probes
    n_probe: int = 300  # probe strings (mirror structure, fixed per learner)


def run_nrt(cfg: NRTConfig, cur: NRTCurriculum, seed: int, control: bool = False) -> dict:
    """One learner through the curriculum. Arrays are per trial unless noted."""
    rng = np.random.default_rng(seed)
    learner = NRTLearner(cfg, seed)
    instruct(learner, cur, rng)
    d_pre, r_pre = make_nrt_strings(cur.n_pre, rng, mirror=False)
    d_main, r_main = make_nrt_strings(cur.n_main, rng, mirror=not control)
    digits = torch.as_tensor(digit_index(np.concatenate([d_pre, d_main])))
    resps = torch.as_tensor(digit_index(np.concatenate([r_pre, r_main])))
    d_probe, _ = make_nrt_strings(cur.n_probe, rng, mirror=True)
    probe = torch.as_tensor(digit_index(d_probe))

    n = len(digits)
    answer_step = np.empty(n, dtype=int)
    correct = np.empty(n, dtype=bool)
    early = np.empty(n, dtype=bool)
    probes: dict[str, list] = {k: [] for k in
                               ("trial", "twonn", "corrdim", "erank", "gate_sparsity")}
    b = cfg.batch
    for i in range(0, n, b):
        if i % cur.probe_every == 0:
            _probe(learner, probe, i, probes)
        s, c, e = learner.online_trial(digits[i : i + b], resps[i : i + b], rng)
        answer_step[i : i + b], correct[i : i + b], early[i : i + b] = s, c, e
    _probe(learner, probe, n, probes)
    return dict(
        answer_step=answer_step,
        correct=correct,
        early=early,
        early_correct=early & correct,
        onset=cur.n_pre,
        probes={k: np.asarray(v) for k, v in probes.items()},
        gates=learner.net.gate.detach().numpy().copy(),
    )


def instruct(learner: NRTLearner, cur: NRTCurriculum, rng: np.random.Generator) -> None:
    """Teach the step rules (Adam, strings without structure)."""
    opt = torch.optim.Adam(learner.net.parameters(), lr=cur.instruct_lr, foreach=False)
    for _ in range(cur.n_instruct):
        d, r = make_nrt_strings(cur.instruct_batch, rng, mirror=False)
        loss, _ = nrt_loss(learner.net, torch.as_tensor(digit_index(d)),
                           torch.as_tensor(digit_index(r)), learner.cfg.final_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def _probe(learner: NRTLearner, probe: torch.Tensor, trial: int, out: dict) -> None:
    h = learner.hidden_states(probe)[:, EARLY_STEP - 1].numpy().astype(float)
    out["trial"].append(trial)
    out["twonn"].append(twonn_dimension(h))
    out["corrdim"].append(correlation_dimension(h))
    out["erank"].append(effective_rank(h))
    out["gate_sparsity"].append(gate_sparsity(learner.net.gate.detach().numpy(), 1e-3))
