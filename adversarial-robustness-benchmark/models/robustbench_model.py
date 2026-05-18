"""RobustBench ceiling-model wrapper (Phase 0).

Wraps an adversarially-trained model from the RobustBench zoo (default:
``Salman2020Do_R50``, Linf, ImageNet) behind the same BaseClassifier interface.
It is a robustness-ceiling reference point, not a benchmark contender.

Requires the ``robustbench`` package:
    pip install --no-deps -r requirements-attacks.txt
"""

from __future__ import annotations

import torch

from .base import BaseClassifier


class RobustBenchClassifier(BaseClassifier):
    """BaseClassifier wrapper around a RobustBench zoo model."""

    def __init__(
        self,
        name: str = "Salman2020Do_R50",
        device: str = "cpu",
        threat_model: str = "Linf",
    ):
        from robustbench.utils import load_model
        from torchvision import transforms as T
        from torchvision.models import ResNet50_Weights

        self.name = name
        self.device = device
        # RobustBench ImageNet models bundle their own input normalization and
        # expect inputs already in [0, 1] pixel space.
        self.model = (
            load_model(model_name=name, dataset="imagenet", threat_model=threat_model)
            .eval()
            .to(device)
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.categories = list(ResNet50_Weights.DEFAULT.meta["categories"])
        self._preprocess = T.Compose(
            [
                T.Resize(256, antialias=True),
                T.CenterCrop(224),
                T.ToTensor(),  # (3, H, W) float in [0, 1]
            ]
        )

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    def logits(self, batch_0_1):
        # No separate normalization: the RobustBench model handles it internally.
        return self.model(batch_0_1.to(self.device))

    def predict(self, image):
        x = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            probs = self.logits(x).softmax(dim=1)
        conf, idx = probs.max(dim=1)
        return self.categories[int(idx)], float(conf)
