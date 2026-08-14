"""Tests for runner output helpers."""

import pandas as pd

from runner import target_feature_table


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
