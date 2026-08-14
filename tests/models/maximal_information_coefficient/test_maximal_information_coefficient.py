"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Behavioral tests for the MIC implementation.
"""

import numpy as np
import pandas as pd
import pytest

from models.maximal_information_coefficient import (
    maximal_information_coefficient,
    maximal_information_coefficient_matrix,
)


def test_identity_and_linear_relationship_are_one() -> None:
    x = np.arange(200, dtype=float)

    assert maximal_information_coefficient(x, x) == pytest.approx(1.0)
    assert maximal_information_coefficient(x, 3 * x + 7) == pytest.approx(1.0)


def test_detects_nonlinear_relationship() -> None:
    x = np.linspace(-1, 1, 500)
    y = x**2

    assert maximal_information_coefficient(x, y) > 0.8


def test_dependence_exceeds_permuted_data() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 600)
    y = np.sin(3 * x)

    assert maximal_information_coefficient(x, y) > maximal_information_coefficient(
        x, rng.permutation(y)
    )


def test_score_is_symmetric_and_bounded() -> None:
    x = np.arange(100)
    y = (x % 7) ** 2
    xy = maximal_information_coefficient(x, y)

    assert 0 <= xy <= 1
    assert xy == pytest.approx(maximal_information_coefficient(y, x))


def test_constant_variable_has_zero_score() -> None:
    assert maximal_information_coefficient([1] * 20, range(20)) == 0.0


def test_missing_values_can_be_dropped_or_rejected() -> None:
    x = [0, 1, 2, None, 4]
    y = [0, 1, 2, 3, 4]

    assert np.isfinite(maximal_information_coefficient(x, y))
    with pytest.raises(ValueError, match="missing"):
        maximal_information_coefficient(x, y, missing="raise")


def test_matrix_preserves_dataframe_labels_and_is_symmetric() -> None:
    frame = pd.DataFrame({"x": range(100), "y": np.arange(100) ** 2})
    result = maximal_information_coefficient_matrix(frame)

    assert list(result.index) == ["x", "y"]
    assert list(result.columns) == ["x", "y"]
    assert np.allclose(result, result.T)
    assert np.allclose(np.diag(result), 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [({"alpha": 0}, "alpha"), ({"alpha": 1.1}, "alpha"), ({"max_bins": 3}, "max_bins")],
)
def test_invalid_parameters_raise(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        maximal_information_coefficient([0, 1], [0, 1], **kwargs)
