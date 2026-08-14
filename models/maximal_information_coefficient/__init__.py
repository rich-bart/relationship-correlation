"""
Copyright (c) 2026 Richard Bartlewitz. All Rights Reserved.
Author: Richard Bartlewitz
Creation: August 2026
Purpose: Public interface for the maximal-information-coefficient model.
"""

from models.maximal_information_coefficient.characteristic_matrix import (
    characteristic_matrix,
)
from models.maximal_information_coefficient.maximal_information_coefficient import (
    maximal_information_coefficient,
    maximal_information_coefficient_matrix,
    mic_matrix,
)

__all__ = [
    "characteristic_matrix",
    "maximal_information_coefficient",
    "maximal_information_coefficient_matrix",
    "mic_matrix",
]
