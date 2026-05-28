"""DefenseModel — adversarially fine-tuned ResNet-50 (Phase 4).

Architecture is identical to torchvision's pretrained ResNet-50. The only
difference is the weights — fine-tuned with Madry-style PGD adversarial
training by ``scripts/train_defense.py``.

By subclassing ``TorchVisionClassifier`` we inherit the full preprocessing
pipeline (resize/crop/to-tensor) and the [0,1]-pixel-space ``logits`` contract
that every attack in this repo assumes. After construction, the classifier
behaves exactly like the baseline ``resnet50`` entry — same eval mode, same
frozen-gradient default — so the existing benchmark runners can evaluate it
through the same code path with just ``--models defense_resnet50``.

The checkpoint path resolves in this order:
  1. ``checkpoint_path`` constructor argument (explicit override),
  2. ``$DEFENSE_CHECKPOINT_PATH`` environment variable,
  3. the default ``models/checkpoints/defense_final.pt`` relative to the repo.

A friendly error is raised if no checkpoint can be found, since the rest of the
pipeline is useless without the trained weights.
"""

from __future__ import annotations

import os

import torch

from .classifiers import TorchVisionClassifier

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CHECKPOINT = os.path.join(_REPO_ROOT, "models", "checkpoints", "defense_final.pt")


class DefenseModel(TorchVisionClassifier):
    """ResNet-50 architecture with adversarially fine-tuned weights."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        # Build the same wrapper as the baseline `resnet50` — this loads the
        # standard pretrained weights, the right preprocess pipeline, the
        # right mean/std, and registers `self.model` as a torchvision ResNet-50.
        super().__init__("resnet50", device=device)
        self.name = "defense_resnet50"

        resolved = (
            checkpoint_path
            or os.environ.get("DEFENSE_CHECKPOINT_PATH")
            or _DEFAULT_CHECKPOINT
        )
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Defense checkpoint not found at {resolved}. "
                "Train it via `python scripts/train_defense.py`, or set "
                "DEFENSE_CHECKPOINT_PATH to point at an existing checkpoint."
            )

        try:
            state = torch.load(resolved, map_location=device, weights_only=True)
        except TypeError:  # older torch
            state = torch.load(resolved, map_location=device)

        weights = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Checkpoint at {resolved} has unexpected keys: {unexpected[:5]}..."
            )
        if missing:
            raise RuntimeError(
                f"Checkpoint at {resolved} is missing keys: {missing[:5]}..."
            )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.checkpoint_path = resolved
