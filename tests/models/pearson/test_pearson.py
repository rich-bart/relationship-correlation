"""Tests for Pearson correlation."""

import numpy as np
import pandas as pd
import pytest

from models.pearson import pearson_correlation, pearson_correlation_matrix


def test_perfect_positive_and_negative_relationships() -> None:
    x = np.arange(20)

    assert pearson_correlation(x, 2 * x + 1) == pytest.approx(1.0)
    assert pearson_correlation(x, -3 * x + 4) == pytest.approx(-1.0)


def test_constant_input_is_undefined() -> None:
    assert np.isnan(pearson_correlation([1, 1, 1], [1, 2, 3]))


def test_matrix_is_labeled_and_symmetric() -> None:
    data = pd.DataFrame({"x": range(10), "y": range(10, 20)})
    result = pearson_correlation_matrix(data)

    assert list(result.columns) == ["x", "y"]
    assert np.allclose(result, result.T)
    assert result.loc["x", "y"] == pytest.approx(1.0)


def test_pairwise_missing_values_are_dropped() -> None:
    assert pearson_correlation([1, 2, None, 4], [2, 4, 10, 8]) == pytest.approx(1.0)
