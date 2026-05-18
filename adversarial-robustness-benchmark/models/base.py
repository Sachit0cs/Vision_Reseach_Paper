"""BaseClassifier interface (project brief, Section 5.3).

Key design rule: ``preprocess`` produces tensors in [0, 1] pixel space, and
normalization is a SEPARATE step done inside ``logits``. This lets adversarial
attacks bound epsilon in pixel space while still differentiating through the
model.

This module defines the interface only. Concrete wrappers (Phase 0) live in
``classifiers.py`` and ``robustbench_model.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Common interface every benchmarked classifier implements."""

    name: str
    device: str
    categories: list  # class names, index-aligned to logits
    model: object     # underlying torch.nn.Module, in eval mode

    @abstractmethod
    def preprocess(self, image):
        """PIL.Image -> Tensor (3, H, W) in [0, 1]. Resize + center-crop only.

        NO normalization here — normalization belongs in ``logits``.
        """
        raise NotImplementedError

    @abstractmethod
    def logits(self, batch_0_1):
        """Tensor (B, 3, H, W) in [0, 1] -> raw logits.

        Internally applies ``(x - mean) / std`` then the network. MUST stay
        differentiable w.r.t. the input so white-box attacks work.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image):
        """PIL.Image -> (label: str, confidence: float). Convenience method."""
        raise NotImplementedError
