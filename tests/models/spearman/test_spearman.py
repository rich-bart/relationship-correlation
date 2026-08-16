"""Tests for Spearman rank correlation."""

import numpy as np
import pandas as pd
import pytest

from models.spearman import spearman_correlation, spearman_correlation_matrix


def test_detects_perfect_nonlinear_monotonic_relationship() -> None:
    x = np.arange(1, 30, dtype=float)

    assert spearman_correlation(x, x**3) == pytest.approx(1.0)
    assert spearman_correlation(x, -(x**3)) == pytest.approx(-1.0)


def test_ties_receive_average_ranks() -> None:
    x = [1, 1, 2, 2, 3, 3]
    y = [10, 10, 20, 20, 30, 30]

    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_nonmonotonic_relationship_is_not_perfect() -> None:
    x = np.linspace(-2, 2, 101)

    assert abs(spearman_correlation(x, x**2)) < 0.1


def test_matrix_is_labeled_symmetric_and_handles_missing_values() -> None:
    data = pd.DataFrame({"x": [1, 2, None, 4], "y": [2, 4, 8, 8]})
    result = spearman_correlation_matrix(data)

    assert list(result.columns) == ["x", "y"]
    assert np.allclose(result, result.T, equal_nan=True)
    assert result.loc["x", "y"] == pytest.approx(1.0)
