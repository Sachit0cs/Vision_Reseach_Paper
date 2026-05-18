"""Typographic overlay attack (Phase 1 — STUB).

Render misleading class-name text onto images with PIL (configurable position,
size, opacity). Model-agnostic. Underexplored on pure classifiers — a likely
source of the paper's novelty (the CLIP / language-grounding hypothesis).
"""

from __future__ import annotations

from .base import BaseAttack


class TypographicOverlay(BaseAttack):
    name = "typographic"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement typographic overlay.")
