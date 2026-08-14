"""Tests for runner output helpers."""

import pandas as pd

from runner import add_permutation_p_values, target_feature_table


def test_target_feature_table_is_ranked_and_excludes_target() -> None:
    matrix = pd.DataFrame(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]],
        index=["target", "strong", "weak"],
        columns=["target", "strong", "weak"],
    )

    result = target_feature_table(matrix, "target")

    assert result["Feature"].tolist() == ["strong", "weak"]
    assert result["Target"].tolist() == ["target", "target"]
    assert result["Coefficient"].tolist() == [0.8, 0.2]


def test_permutation_p_values_are_reproducible_and_bounded() -> None:
    data = pd.DataFrame(
        {
            "x": [0, 0, 1, 1] * 20,
            "y": [0, 0, 1, 1] * 20,
        }
    )
    pairs = pd.DataFrame(
        [("x", "y", 1.0)],
        columns=["Feature 1", "Feature 2", "Coefficient"],
    )
    config = {
        "permutations": 19,
        "random_seed": 7,
        "discrete": True,
        "analysis": "mutual_information",
        "bins": "auto",
        "normalize": True,
        "base": 2.0,
    }

    first = add_permutation_p_values(
        pairs, data, config, ("Feature 1", "Feature 2")
    )
    second = add_permutation_p_values(
        pairs, data, config, ("Feature 1", "Feature 2")
    )

    assert first["P-value"].tolist() == second["P-value"].tolist()
    assert first.loc[0, "P-value"] == 0.05
