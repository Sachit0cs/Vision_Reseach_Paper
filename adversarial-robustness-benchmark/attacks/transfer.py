"""Surrogate-based transfer attack harness (Phase 1 — STUB).

Generate adversarial images on a fixed surrogate model, then hand them to the
evaluation pipeline for testing on every model. Measures cross-architecture
transferability — reported separately from white-box results (Section 3.3).
"""

from __future__ import annotations


class TransferHarness:
    """Crafts adversarial images on a surrogate, evaluates them on all models."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 1: implement the transfer harness.")
