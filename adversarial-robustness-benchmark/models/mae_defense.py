"""MAE_defense_model — frozen-MAE input purifier in front of ResNet-50.

A TEST-TIME defense (NO training): prediction = resnet50( purify(x) ), where
``purify`` uses a frozen pretrained ``facebook/vit-mae-base`` to mask ~mask_ratio
of the 16x16 patches and reconstruct them, averaging over K random masks (an
ensemble that cleanses adversarial perturbation). The ResNet-50 classifier is
NEVER modified or retrained — this is purely a front-end input transformation.

Wired as the benchmark model key ``MAE_defense_model`` so the existing attack /
benchmark runners evaluate it through the same [0,1]-pixel-space ``logits``
contract every baseline uses (normalization happens inside ``logits``).

ADAPTIVE-ATTACK HONESTY (the obfuscated-gradients trap)
-------------------------------------------------------
The MAE reconstruction runs under ``torch.no_grad`` (it is effectively
non-differentiable, and is also stochastic via random masking). If it were wired
naively into ``logits``, the gradient attacks (FGSM/PGD/APGD) would either crash
(the loss is disconnected from the input) or report falsely-inflated robustness
because gradients cannot flow through the purifier — the classic "obfuscated
gradients" artifact (Athalye et al., 2018).

To give the gradient attacks an HONEST signal, ``logits`` uses a BPDA
(Backward-Pass Differentiable Approximation) straight-through estimator: the
classifier SEES the purified image in the forward pass, but the gradient flows
through the purifier as the identity (purify(x) ~= x). This is the standard
adaptive attack against input-transformation defenses. The gradient-FREE Square
attack additionally queries the true stochastic pipeline directly. Together they
are a credible adaptive evaluation — a purifier that is merely masking gradients
will be exposed by BPDA + Square rather than flattered by them.

Config (read once at construction; override via env for tractability):
  MAE_MASK_RATIO  (default 0.5)  — fraction of patches masked per reconstruction.
  MAE_K           (default 8)    — ensemble size (averaged reconstructions).
  MAE_CHUNK       (default 32)   — MAE sub-batch size (lower if you OOM).
  MAE_ID          (default facebook/vit-mae-base) — HF model id.
"""

from __future__ import annotations

import os

import torch

from .classifiers import TorchVisionClassifier

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MAEDefenseModel(TorchVisionClassifier):
    """ResNet-50 with a frozen-MAE input purifier in front (test-time defense)."""

    def __init__(
        self,
        device: str = "cpu",
        mask_ratio: float | None = None,
        k: int | None = None,
        chunk: int | None = None,
        mae_id: str | None = None,
    ):
        # Build the baseline ResNet-50 wrapper: pretrained weights, the right
        # preprocess pipeline, the right mean/std, and the [0,1] logits contract.
        super().__init__("resnet50", device=device)
        self.name = "MAE_defense_model"

        self.mask_ratio = float(mask_ratio if mask_ratio is not None
                                else os.environ.get("MAE_MASK_RATIO", 0.5))
        self.k = int(k if k is not None else os.environ.get("MAE_K", 8))
        self.chunk = int(chunk if chunk is not None else os.environ.get("MAE_CHUNK", 32))
        self.mae_id = mae_id or os.environ.get("MAE_ID", "facebook/vit-mae-base")

        from transformers import ViTMAEForPreTraining

        self.mae = ViTMAEForPreTraining.from_pretrained(self.mae_id).eval().to(device)
        for p in self.mae.parameters():
            p.requires_grad_(False)
        self.norm_pix_loss = bool(getattr(self.mae.config, "norm_pix_loss", False))

        self._mae_mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self._mae_std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    # --- MAE purification (identical math to mae_smoke/mae_purifier.py) -------

    def _mae_norm(self, x01):
        return (x01 - self._mae_mean) / self._mae_std

    def _mae_denorm(self, xn):
        return xn * self._mae_std + self._mae_mean

    @torch.no_grad()
    def _reconstruct_once(self, pix_norm):
        self.mae.config.mask_ratio = self.mask_ratio
        out = self.mae(pixel_values=pix_norm)
        pred = out.logits
        mask = out.mask
        orig_patches = self.mae.patchify(pix_norm)
        if self.norm_pix_loss:
            pm = orig_patches.mean(dim=-1, keepdim=True)
            pv = orig_patches.var(dim=-1, keepdim=True)
            pred = pred * (pv + 1e-6) ** 0.5 + pm
        m = mask.unsqueeze(-1)
        combined = orig_patches * (1 - m) + pred * m
        return self.mae.unpatchify(combined)

    @torch.no_grad()
    def _purify(self, x01):
        """K-ensemble masked reconstruction. Returns a DETACHED [0,1] tensor."""
        x01 = x01.to(self.device).float()
        n = max(1, self.k)
        outs = []
        for s in range(0, x01.size(0), self.chunk):
            pix = self._mae_norm(x01[s:s + self.chunk])
            acc = torch.zeros_like(pix)
            for _ in range(n):
                acc = acc + self._reconstruct_once(pix)
            outs.append(self._mae_denorm(acc / n).clamp(0.0, 1.0))
        return torch.cat(outs, dim=0)

    # --- the [0,1] -> logits contract, with BPDA straight-through ------------

    def logits(self, batch_0_1):
        x = batch_0_1.to(self.device).float()
        x_pure = self._purify(x)                  # detached, [0,1]
        # BPDA straight-through: forward value = purified image; gradient w.r.t.
        # the (attacked) input flows through the purifier as the identity.
        x_bpda = x + (x_pure - x).detach()
        return super().logits(x_bpda)
