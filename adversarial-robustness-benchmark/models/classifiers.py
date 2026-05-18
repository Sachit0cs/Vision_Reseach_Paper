"""The 7 benchmarked architecture wrappers + factory (Phase 0 — STUB).

To implement:
  * TorchVisionClassifier — ResNet-50, VGG-16, ConvNeXt-Tiny, ViT-B/16,
    Swin-Tiny, EfficientNet-B0 from torchvision.models with DEFAULT weights.
    Split torchvision's bundled transform into a [0,1] preprocess plus a
    separate Normalize(mean, std) applied inside `logits`.
  * ClipZeroShotClassifier — CLIP (ViT-B/16). Precompute text features once
    for all class names with the prompt "a photo of a {class}"; `logits`
    projects normalized image features against them, scaled by the logit
    scale. Stays differentiable so white-box attacks work.
  * build_classifier(name, device) — factory keyed by the short names in
    config.yaml (resnet50, vgg16, convnext_tiny, vit_b_16, swin_t,
    efficientnet_b0, clip_vit_b16).
"""

from __future__ import annotations

from .base import BaseClassifier


class TorchVisionClassifier(BaseClassifier):
    """Wrapper for torchvision ImageNet-pretrained CNNs/Transformers."""

    def __init__(self, name: str, device: str = "cpu"):
        raise NotImplementedError("Phase 0: implement torchvision wrappers.")


class ClipZeroShotClassifier(BaseClassifier):
    """CLIP (ViT-B/16) used as a zero-shot ImageNet classifier."""

    def __init__(self, device: str = "cpu"):
        raise NotImplementedError("Phase 0: implement CLIP zero-shot wrapper.")


def build_classifier(name: str, device: str = "cpu") -> BaseClassifier:
    """Factory: short model key -> BaseClassifier instance."""
    raise NotImplementedError("Phase 0: implement the classifier factory.")
