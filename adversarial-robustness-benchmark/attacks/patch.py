"""Adversarial patch attack (Phase 1 — STUB).

Optimize or place an adversarial sticker onto the image. A fixed universal
patch produces a model-agnostic poisoned set (part of the shared poisoned set).
"""

from __future__ import annotations

from .base import BaseAttack


class AdversarialPatch(BaseAttack):
    name = "patch"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement adversarial patch.")
