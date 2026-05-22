"""Classification robustness metrics.

Per (model, attack):
  * clean accuracy      — accuracy on clean images.
  * robust accuracy     — accuracy on attacked images (the headline number).
  * accuracy drop       — clean minus robust.
  * fooling / flip rate — fraction of originally-correct predictions that flip.
  * CDS                 — Confidence Degradation Score: mean drop in the
                          softmax probability of the predicted class.
  * softmax KL          — mean KL divergence between clean and attacked softmax.

Only the two headline metrics (`clean_accuracy`, `robust_accuracy`) are
implemented here. The rest are reserved for Phase 2; the gradient-attack runner
only needs the two below.
"""

from __future__ import annotations

import torch


def _predict_labels(classifier, batch_0_1: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """Return argmax labels (long) for a [0,1] image batch, mini-batched."""
    device = classifier.device
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batch_0_1.shape[0], batch_size):
            chunk = batch_0_1[start : start + batch_size].to(device)
            logits = classifier.logits(chunk)
            preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).long()


def clean_accuracy(
    classifier,
    images_0_1: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Top-1 accuracy of ``classifier`` on a batch of clean [0,1] images."""
    labels = torch.as_tensor(labels).long()
    preds = _predict_labels(classifier, images_0_1, batch_size=batch_size)
    return (preds == labels).float().mean().item()


def robust_accuracy(
    classifier,
    adv_images_0_1: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Top-1 accuracy on adversarial images — the headline robustness number."""
    labels = torch.as_tensor(labels).long()
    preds = _predict_labels(classifier, adv_images_0_1, batch_size=batch_size)
    return (preds == labels).float().mean().item()


def label_flip_rate(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement label-flip rate.")


def confidence_degradation_score(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement CDS.")


def softmax_kl_divergence(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement softmax KL-divergence.")
