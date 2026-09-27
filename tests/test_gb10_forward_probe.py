from types import SimpleNamespace

import numpy as np
import pytest

from tools.gb10.forward_probe import validate_action


@pytest.mark.parametrize("dimension", [19, 38])
def test_finite_chunk(dimension):
    chunk = np.zeros((16, dimension), dtype=np.float32)
    result = validate_action(SimpleNamespace(action_chunk=chunk, latency_ms=5), dimension)
    assert result is chunk


@pytest.mark.parametrize("shape", [(19,), (0, 19), (16, 18), (1, 1, 19)])
def test_bad_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        validate_action(SimpleNamespace(action_chunk=np.zeros(shape), latency_ms=5), 19)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_action(value):
    chunk = np.zeros((16, 19))
    chunk[0, 0] = value
    with pytest.raises(ValueError, match="Non-finite"):
        validate_action(SimpleNamespace(action_chunk=chunk, latency_ms=5), 19)


@pytest.mark.parametrize("latency", [-1, np.nan, np.inf])
def test_invalid_latency(latency):
    with pytest.raises(ValueError, match="latency"):
        validate_action(SimpleNamespace(action_chunk=np.zeros((16, 19)), latency_ms=latency), 19)
