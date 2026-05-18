"""Classification robustness metrics (Phase 2 — STUB).

Per (model, attack):
  * clean accuracy      — accuracy on clean images.
  * robust accuracy     — accuracy on attacked images (the headline number).
  * accuracy drop       — clean minus robust.
  * fooling / flip rate — fraction of originally-correct predictions that flip.
  * CDS                 — Confidence Degradation Score: mean drop in the
                          softmax probability of the predicted class.
  * softmax KL          — mean KL divergence between clean and attacked softmax.
"""

from __future__ import annotations


def clean_accuracy(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement clean accuracy.")


def robust_accuracy(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement robust accuracy.")


def label_flip_rate(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement label-flip rate.")


def confidence_degradation_score(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement CDS.")


def softmax_kl_divergence(*args, **kwargs):
    raise NotImplementedError("Phase 2: implement softmax KL-divergence.")
