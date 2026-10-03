import json

from nautilus.harness import cached, file_sha256, run_parallel, save_json, seed_for


def test_seed_for_stable_and_distinct():
    assert seed_for("a", 1) == seed_for("a", 1)
    assert seed_for("a", 1) != seed_for("a", 2)
    assert 0 <= seed_for("x") < 2**63


def test_save_json_hash_and_cache(tmp_path):
    p = tmp_path / "r.json"
    digest = save_json(p, {"b": 1, "a": [1.5]})
    assert digest == file_sha256(p)
    assert json.loads(p.read_text()) == {"a": [1.5], "b": 1}
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1}

    q = tmp_path / "c.json"
    assert cached(q, compute) == {"x": 1}
    assert cached(q, compute) == {"x": 1}
    assert len(calls) == 1


def _square(x):
    return x * x


def test_run_parallel_serial_path():
    assert run_parallel(_square, [1, 2, 3], workers=1) == [1, 4, 9]
