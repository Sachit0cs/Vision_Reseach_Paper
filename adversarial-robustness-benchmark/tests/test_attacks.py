"""Attack sanity checks (Phase 1 — STUB).

Verify (project brief, Section 7):
  * every attack measurably reduces robust accuracy on a known-weak model (VGG-16);
  * AutoAttack robust accuracy <= PGD robust accuracy for the same model;
  * Square (black-box) does not dramatically outperform PGD (no gradient masking).
"""

import pytest


@pytest.mark.skip(reason="Phase 1: attacks not implemented yet.")
def test_pgd_reduces_vgg16_accuracy():
    raise NotImplementedError
