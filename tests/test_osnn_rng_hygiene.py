"""RNG hygiene and seed reproducibility for the OSNN SSL classifier.

OSNN used to reseed the process-global ``random``, ``numpy`` and ``torch``
RNGs in its constructor. That silently hijacked the caller's random
streams (and coupled unrelated learners that share those streams). The
tests below pin the intended invariant: constructing OSNN must leave the
global RNGs untouched, and the same ``seed`` must reproduce the same
training trajectory.
"""

import random

import numpy as np
import pytest

from capymoa.anomaly.datasets import TinyBlobs

torch = pytest.importorskip("torch")

from capymoa.ssl import OSNN


def _make(osnn_seed, optim_steps=10):
    stream = TinyBlobs()
    return OSNN(schema=stream.get_schema(), seed=osnn_seed, optim_steps=optim_steps)


def test_constructor_leaves_global_torch_rng_untouched():
    torch.manual_seed(7)
    _make(1)
    after = torch.rand(3)
    torch.manual_seed(7)
    expected = torch.rand(3)
    assert torch.equal(after, expected)


def test_constructor_leaves_global_numpy_rng_untouched():
    np.random.seed(7)
    _make(1)
    after = np.random.rand(3)
    np.random.seed(7)
    expected = np.random.rand(3)
    assert np.array_equal(after, expected)


def test_constructor_leaves_global_python_rng_untouched():
    random.seed(7)
    _make(1)
    after = random.random()
    random.seed(7)
    expected = random.random()
    assert after == expected


def test_same_seed_same_trajectory():
    def param_sums():
        model = _make(3)
        stream = TinyBlobs()
        for n, instance in enumerate(stream):
            model.train(instance)
            if n >= 120:
                break
        return [
            (float(p.sum()), float(p.abs().sum())) for p in model.Network.parameters()
        ]

    assert param_sums() == param_sums()
