import numpy as np
import pytest

from ally_fall.weights import class_weights


def test_weights_follow_effective_number() -> None:
    w = class_weights(np.array([100, 10]), 0.99)
    expected = (1 - 0.99**100) / (1 - 0.99**10)  # weight ratio = inverse ratio of effective numbers
    assert abs(w[1] / w[0] - expected) < 1e-5


def test_beta_zero_gives_equal_weights() -> None:
    np.testing.assert_allclose(class_weights(np.array([500, 3, 40]), 0.0), [1.0, 1.0, 1.0])


def test_empty_class_weight_zero() -> None:
    w = class_weights(np.array([50, 0, 5]), 0.999)
    assert w[1] == 0.0 and np.isfinite(w).all()
    assert abs(float(w[[0, 2]].mean()) - 1.0) < 1e-6


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_rejects_beta_out_of_range(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        class_weights(np.array([1, 2]), beta)
