"""Verification tests for the mutual-information model."""

import numpy as np
import pandas as pd
import pytest

from models.mutual_information import (
    mutual_information,
    mutual_information_matrix,
)


def test_identical_balanced_binary_variables_equal_one_bit() -> None:
    values = [0, 0, 1, 1]

    assert mutual_information(values, values) == pytest.approx(1.0)


def test_independent_balanced_binary_variables_equal_zero() -> None:
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    assert mutual_information(x, y) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_is_symmetric_and_nonnegative() -> None:
    x = ["a", "a", "b", "b", "c", "c"]
    y = [1, 1, 1, 2, 2, 2]

    xy = mutual_information(x, y)
    yx = mutual_information(y, x)

    assert xy >= 0
    assert xy == pytest.approx(yx)


@pytest.mark.parametrize("normalization", [True, "sqrt", "min", "max"])
def test_normalized_identity_is_one(normalization) -> None:
    values = [0, 0, 1, 1, 2, 2]

    score = mutual_information(values, values, normalize=normalization)

    assert score == pytest.approx(1.0)


def test_normalized_result_is_bounded() -> None:
    x = [0, 0, 0, 1, 1, 1]
    y = [0, 0, 1, 0, 1, 1]

    score = mutual_information(x, y, normalize=True)

    assert 0.0 <= score <= 1.0


def test_normalized_constant_variable_returns_zero() -> None:
    constant = [1, 1, 1, 1]

    assert mutual_information(constant, constant, normalize=True) == 0.0


def test_matrix_is_labeled_symmetric_and_has_expected_values() -> None:
    data = pd.DataFrame(
        {
            "identical_a": [0, 0, 1, 1],
            "identical_b": [0, 0, 1, 1],
            "independent": [0, 1, 0, 1],
        }
    )

    matrix = mutual_information_matrix(
        data,
        discrete=True,
        normalize=True,
    )

    assert isinstance(matrix, pd.DataFrame)
    assert list(matrix.index) == list(data.columns)
    assert list(matrix.columns) == list(data.columns)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix.loc["identical_a", "identical_b"] == pytest.approx(1.0)
    assert matrix.loc["identical_a", "independent"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_pairwise_missing_values_match_explicitly_dropped_values() -> None:
    x = [0, 0, 1, 1, None]
    y = [0, 0, 1, 1, 0]

    with_missing = mutual_information(x, y, missing="drop")
    explicitly_dropped = mutual_information(x[:-1], y[:-1])

    assert with_missing == pytest.approx(explicitly_dropped)


def test_raise_missing_policy_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="missing"):
        mutual_information([0, None, 1], [0, 1, 1], missing="raise")


def test_matrix_listwise_missing_policy_returns_finite_matrix() -> None:
    data = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, np.nan],
            "b": [0, 0, 1, 1, 0],
            "c": [1, 0, 1, 0, 1],
        }
    )

    matrix = mutual_information_matrix(
        data,
        discrete=True,
        missing="listwise",
    )

    assert matrix.shape == (3, 3)
    assert np.isfinite(matrix.to_numpy()).all()


def test_dependent_continuous_pair_exceeds_shuffled_pair() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=2_000)
    dependent = x + rng.normal(scale=0.1, size=len(x))
    shuffled = rng.permutation(dependent)

    dependent_score = mutual_information(
        x,
        dependent,
        discrete_x=False,
        discrete_y=False,
    )
    shuffled_score = mutual_information(
        x,
        shuffled,
        discrete_x=False,
        discrete_y=False,
    )

    assert dependent_score > shuffled_score


def test_continuous_dependence_exceeds_permutation_null_distribution() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=1_000)
    dependent = np.sin(x) + rng.normal(scale=0.05, size=len(x))

    observed = mutual_information(
        x,
        dependent,
        discrete_x=False,
        discrete_y=False,
    )
    null_scores = [
        mutual_information(
            x,
            rng.permutation(dependent),
            discrete_x=False,
            discrete_y=False,
        )
        for _ in range(40)
    ]

    assert observed > np.percentile(null_scores, 95)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        ([0, 1], [0], "same length"),
        ([0, 1], [0, 1], "base"),
    ],
)
def test_invalid_inputs_raise_clear_errors(x, y, message) -> None:
    arguments = {"base": 1.0} if message == "base" else {}

    with pytest.raises(ValueError, match=message):
        mutual_information(x, y, **arguments)
