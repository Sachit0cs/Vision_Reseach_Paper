"""White-box / black-box gradient attacks (Phase 1 — STUB).

  * FGSM  — single-step L-infinity, weak baseline.
  * PGD   — multi-step L-infinity (epsilon 8/255, step 2/255, 10-20 iters).
  * AutoAttack — gold-standard ensemble; use the official `autoattack` package.
  * Square — score-based black-box attack; use `torchattacks`.

All operate in [0, 1] space. FGSM/PGD may be implemented directly or via
`torchattacks`.
"""

from __future__ import annotations

from .base import BaseAttack


class FGSM(BaseAttack):
    name = "fgsm"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement FGSM.")


class PGD(BaseAttack):
    name = "pgd"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement PGD.")


class AutoAttackWrapper(BaseAttack):
    name = "autoattack"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement AutoAttack wrapper.")


class SquareAttack(BaseAttack):
    name = "square"

    def apply(self, classifier, image_batch_0_1, labels):
        raise NotImplementedError("Phase 1: implement Square attack.")
