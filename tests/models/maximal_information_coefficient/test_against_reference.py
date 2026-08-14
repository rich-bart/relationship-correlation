"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Reference properties from the MIC definition.
"""

import numpy as np
import pytest

from models.maximal_information_coefficient import characteristic_matrix


def test_characteristic_matrix_normalizes_perfect_binary_grid() -> None:
    x = np.repeat([0.0, 1.0], 50)
    y = x.copy()
    matrix = characteristic_matrix(x, y, budget=4)

    assert matrix[2, 2] == pytest.approx(1.0)


def test_characteristic_matrix_marks_inadmissible_grids() -> None:
    values = np.arange(20, dtype=float)
    matrix = characteristic_matrix(values, values, budget=6)

    assert np.isnan(matrix[3, 3])
    assert np.isfinite(matrix[2, 3])
