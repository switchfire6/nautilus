"""Runs: seeds, single-thread workers, resumable result files, hashes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def seed_for(*keys) -> int:
    """A stable 63-bit seed derived from arbitrary keys (strings, ints, floats)."""
    h = hashlib.sha256(json.dumps([str(k) for k in keys]).encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


def rng_for(*keys) -> np.random.Generator:
    return np.random.default_rng(seed_for(*keys))


def init_worker() -> None:
    """One thread per worker, no denormals, deterministic algorithms."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    torch.use_deterministic_algorithms(True)


def run_parallel(fn: Callable, jobs: Iterable, workers: int = 4) -> list:
    """Map ``fn`` over ``jobs`` in up to ``workers`` single-thread processes, in order."""
    jobs = list(jobs)
    if workers <= 1:
        init_worker()
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"),
                             initializer=init_worker) as ex:
        return list(ex.map(fn, jobs))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path: Path, obj) -> str:
    """Write ``obj`` as sorted JSON and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(obj), indent=1, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)  # atomic, so an interrupted run never leaves a half-written file
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cached(path: Path, compute: Callable[[], dict]) -> dict:
    """Resumability: return the stored result at ``path`` or compute and store it."""
    path = Path(path)
    if path.exists():
        return load_json(path)
    result = compute()
    save_json(path, result)
    return load_json(path)
