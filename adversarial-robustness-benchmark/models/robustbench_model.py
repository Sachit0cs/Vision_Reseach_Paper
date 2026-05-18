"""RobustBench ceiling-model wrapper (Phase 0 — STUB).

Wraps an adversarially-trained model from the RobustBench zoo (default:
``Salman2020Do_R50``, Linf, ImageNet) behind the same BaseClassifier interface.
It is a robustness-ceiling reference point, not a benchmark contender.
"""

from __future__ import annotations

from .base import BaseClassifier


class RobustBenchClassifier(BaseClassifier):
    """BaseClassifier wrapper around a RobustBench zoo model."""

    def __init__(self, name: str = "Salman2020Do_R50", device: str = "cpu"):
        raise NotImplementedError("Phase 0: implement the RobustBench wrapper.")
