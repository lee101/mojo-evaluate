"""Mojo kernels behind a compatible subset of Hugging Face Evaluate."""

from .module import CombinedEvaluations, Metric, combine, load
from .metrics import SUPPORTED_METRICS

__all__ = [
    "CombinedEvaluations",
    "Metric",
    "SUPPORTED_METRICS",
    "combine",
    "load",
]

__version__ = "0.1.0"

