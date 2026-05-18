"""Classifier smoke tests (Phase 0 — STUB).

Verify every one of the 7 models loads, that preprocess returns a [0, 1]
tensor, that logits stays differentiable, and that clean accuracy matches
published ImageNet numbers within ~1%.
"""

import pytest


@pytest.mark.skip(reason="Phase 0: classifier wrappers not implemented yet.")
def test_all_models_load_and_classify():
    raise NotImplementedError
