"""Public interface for the mutual-information model."""

from models.mutual_information.calculator import (
    mutual_information,
    mutual_information_matrix,
)

__all__ = ["mutual_information", "mutual_information_matrix"]
