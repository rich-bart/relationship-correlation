"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: July 2026
Purpose: Compare this project's MI results with scikit-learn.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from models.mutual_information import (
    mutual_information,
    mutual_information_matrix,
)
from models.mutual_information.discretization import _discretize


@pytest.fixture
def categorical_pair() -> tuple[np.ndarray, np.ndarray]:
    """Return two partially dependent categorical variables."""
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    y = np.array([1, 1, 2, 1, 2, 2, 2, 3, 3])
    return x, y


def test_discrete_mi_in_nats_matches_sklearn(categorical_pair) -> None:
    x, y = categorical_pair

    expected = mutual_info_score(x, y)
    actual = mutual_information(x, y, base=np.e)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_discrete_mi_in_bits_matches_converted_sklearn_result(
    categorical_pair,
) -> None:
    x, y = categorical_pair

    expected_bits = mutual_info_score(x, y) / np.log(2)
    actual_bits = mutual_information(x, y, base=2)

    assert actual_bits == pytest.approx(expected_bits, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("our_normalization", "sklearn_average_method"),
    [
        ("sqrt", "geometric"),
        ("min", "min"),
        ("max", "max"),
    ],
)
def test_normalized_mi_matches_sklearn(
    categorical_pair,
    our_normalization,
    sklearn_average_method,
) -> None:
    x, y = categorical_pair

    expected = normalized_mutual_info_score(
        x,
        y,
        average_method=sklearn_average_method,
    )
    actual = mutual_information(x, y, normalize=our_normalization)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_discrete_matrix_matches_sklearn_pair_by_pair() -> None:
    data = pd.DataFrame(
        {
            "alpha": ["a", "a", "b", "b", "c", "c", "c", "a"],
            "beta": [1, 1, 1, 2, 2, 3, 3, 1],
            "gamma": [0, 1, 0, 1, 0, 1, 1, 0],
        }
    )

    actual = mutual_information_matrix(data, discrete=True, base=np.e)

    for left in data.columns:
        for right in data.columns:
            expected = mutual_info_score(data[left], data[right])
            assert actual.loc[left, right] == pytest.approx(
                expected,
                rel=1e-12,
                abs=1e-12,
            )


def test_continuous_histogram_mi_matches_sklearn_on_same_bins() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=1_000)
    y = np.sin(x) + rng.normal(scale=0.1, size=len(x))
    bin_count = 12

    expected = mutual_info_score(
        _discretize(x, bin_count),
        _discretize(y, bin_count),
    )
    actual = mutual_information(
        x,
        y,
        discrete_x=False,
        discrete_y=False,
        bins=bin_count,
        base=np.e,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
