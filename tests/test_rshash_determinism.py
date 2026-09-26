"""Cross-process reproducibility of RSHash anomaly scores.

RSHash derives its sketch hash keys from ``seed``, but the sketch
indexed cells with Python's builtin ``hash()``, which is randomized per
process (PYTHONHASHSEED). The same seed then produced different score
traces in different processes. The test runs a tiny scoring session in
two fresh interpreters with different hash seeds and requires identical
output; it fails on the builtin-hash implementation and passes with the
keyed-digest one.
"""

import os
import subprocess
import sys

from capymoa.anomaly import RSHash
from capymoa.anomaly.datasets import TinyBlobs

PROBE = """
from capymoa.anomaly import RSHash
from capymoa.anomaly.datasets import TinyBlobs

stream = TinyBlobs()
model = RSHash(schema=stream.get_schema(), m=300, s=64, w=4, p=10000, seed=42)
scores = []
for i, instance in enumerate(stream):
    if i >= 200:
        break
    scores.append(model.score_instance(instance))
    model.train(instance)
print(",".join(f"{s:.6f}" for s in scores))
"""


def _run_probe(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout


def _score_trace() -> list[float]:
    stream = TinyBlobs()
    model = RSHash(schema=stream.get_schema(), m=300, s=64, w=4, p=10000, seed=42)
    scores = []
    for i, instance in enumerate(stream):
        if i >= 200:
            break
        scores.append(model.score_instance(instance))
        model.train(instance)
    return scores


def test_scores_stable_across_processes():
    out_a = _run_probe("0")
    out_b = _run_probe("1")
    assert out_a == out_b
    # Sanity: the trace must carry signal, otherwise the comparison is
    # vacuous.
    assert any(s != "0.000000" for s in out_a.strip().split(","))


def test_same_seed_same_scores_in_process():
    assert _score_trace() == _score_trace()


def test_different_seed_different_scores():
    """The seed must actually reach the sketch: two seeds on the same data
    should generally disagree. Guards against a future refactor that makes
    the digest seed-independent (which would also collapse the ensemble
    members' sketch diversity)."""

    def trace(seed):
        stream = TinyBlobs()
        model = RSHash(schema=stream.get_schema(), m=300, s=64, w=4, p=10000, seed=seed)
        scores = []
        for i, instance in enumerate(stream):
            if i >= 200:
                break
            scores.append(model.score_instance(instance))
            model.train(instance)
        return scores

    assert trace(42) != trace(43)
