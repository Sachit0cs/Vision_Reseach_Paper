"""ImageNet-C-style common corruptions (Phase 1 — STUB).

The 15 standard corruption types (noise, blur, weather, digital) at 5
severities. Non-adversarial but a standard robustness axis. Model-agnostic.
"""

from __future__ import annotations

from .base import BaseAttack


class CommonCorruptions(BaseAttack):
    name = "corruptions"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement common corruptions.")
