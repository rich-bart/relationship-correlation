"""Public interface for Spearman rank correlation."""

from models.spearman.spearman import spearman_correlation, spearman_correlation_matrix

__all__ = ["spearman_correlation", "spearman_correlation_matrix"]
