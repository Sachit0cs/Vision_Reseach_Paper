"""Novel defensive architecture (Phase 4 — STUB).

Takes the most robust base architecture from the benchmark and adds ONE
defensive module (project brief, Section 3.4). Candidates:
  * a denoising / feature-purification front-end,
  * a frequency-domain filter (adversarial noise is often high-frequency),
  * an attention-based feature-purification block,
  * a randomized-smoothing layer (certified robustness).

Framing requirement: claim a targeted, measured improvement with an explicit
clean-accuracy / robustness tradeoff. Do NOT claim universal superiority.
"""

from __future__ import annotations

from .base import BaseClassifier


class DefenseModel(BaseClassifier):
    """Base architecture + one defensive module."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 4: implement the defense module.")
