"""MAE-based adversarial input purifier (smoke-test scaffold).

Standalone and DELETABLE. Everything related to the MAE-purifier experiment
lives under ``mae_smoke/``; remove that folder and the repo is untouched.

What this is
------------
A pretrained Masked Autoencoder (``facebook/vit-mae-base``) used as a TEST-TIME
purifier placed IN FRONT of an unmodified classifier::

    prediction = classifier( purify(image) )

This is NOT an architecture change. The classifier is never modified or
retrained; the MAE is a separate, frozen, pretrained network. There is NO
training here.

Mechanism
---------
Mask a large fraction of the 16x16 image patches, let the MAE reconstruct the
masked patches from the visible context, then stitch (visible original patches +
reconstructed masked patches) back into a full image. Adversarial perturbation
living inside a masked patch is discarded and re-synthesised from a clean prior;
averaging over K random maskings spreads that cleansing over the whole image.

Contract
--------
All inputs/outputs are [0,1] pixel-space tensors, shape (B,3,H,W) at the MAE's
native 224x224 — matching the repo's ``preprocess`` -> ``logits`` contract
(see ``models/classifiers.py``). The MAE's own ImageNet normalisation is applied
and undone internally, so callers stay in [0,1].
"""

from __future__ import annotations

import torch

# ImageNet mean/std — the normalisation vit-mae-base was pretrained with.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MAEPurifier:
    """Frozen pretrained-MAE purifier. No grad, no training."""

    def __init__(self, model_id: str = "facebook/vit-mae-base", device: str = "cpu"):
        from transformers import ViTMAEForPreTraining

        self.device = device
        self.model = ViTMAEForPreTraining.from_pretrained(model_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.image_size = int(self.model.config.image_size)
        self.patch_size = int(self.model.config.patch_size)
        # vit-mae-base trains with per-patch-normalised targets; we must invert
        # that to get pixels back. Read the flag rather than assume.
        self.norm_pix_loss = bool(getattr(self.model.config, "norm_pix_loss", False))

        self._mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    # --- normalisation helpers (between [0,1] and the MAE's input space) -------
    def _norm(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def _denorm(self, xn: torch.Tensor) -> torch.Tensor:
        return xn * self._std + self._mean

    @torch.no_grad()
    def _reconstruct_once(self, pix_norm: torch.Tensor, mask_ratio: float):
        """One masked forward pass.

        Returns (purified_norm, mask, recon_full_norm), all in the MAE's
        normalised-pixel space:
          * purified_norm   = visible ORIGINAL patches + reconstructed MASKED patches
          * recon_full_norm = the decoder's reconstruction of ALL patches
          * mask            = (B, L) with 1 = masked, 0 = visible (original order)
        """
        self.model.config.mask_ratio = float(mask_ratio)
        out = self.model(pixel_values=pix_norm)

        pred = out.logits                              # (B, L, p*p*C) target space
        mask = out.mask                                # (B, L), 1 = masked
        orig_patches = self.model.patchify(pix_norm)   # (B, L, p*p*C) normalised-pixel

        if self.norm_pix_loss:
            # ``pred`` is per-patch standardised; de-standardise it with the
            # original patches' per-patch mean/var to land back in pixel space.
            pm = orig_patches.mean(dim=-1, keepdim=True)
            pv = orig_patches.var(dim=-1, keepdim=True)
            pred = pred * (pv + 1e-6) ** 0.5 + pm

        m = mask.unsqueeze(-1)                         # (B, L, 1)
        combined = orig_patches * (1 - m) + pred * m   # keep visible, fill masked
        purified_norm = self.model.unpatchify(combined)
        recon_full_norm = self.model.unpatchify(pred)
        return purified_norm, mask, recon_full_norm

    @torch.no_grad()
    def purify(self, x01: torch.Tensor, n_masks: int = 4, mask_ratio: float = 0.75,
               chunk: int = 32) -> torch.Tensor:
        """Purify a [0,1] batch, averaged over ``n_masks`` random maskings.

        Processed in sub-batches of ``chunk`` so a large run (e.g. n=200 on a
        16 GB T4) does not OOM through the ViT decoder.
        """
        x01 = x01.to(self.device).float()
        n = max(1, int(n_masks))
        outs = []
        for s in range(0, x01.size(0), chunk):
            pix = self._norm(x01[s : s + chunk])
            acc = torch.zeros_like(pix)
            for _ in range(n):
                purified_norm, _, _ = self._reconstruct_once(pix, mask_ratio)
                acc = acc + purified_norm
            outs.append(self._denorm(acc / n).clamp(0.0, 1.0))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def visualize(self, x01: torch.Tensor, mask_ratio: float = 0.75):
        """One random mask, for eyeballing. Returns three [0,1] batches:
        (masked_view, full_reconstruction, purified)."""
        x01 = x01.to(self.device).float()
        pix = self._norm(x01)
        purified_norm, mask, recon_full_norm = self._reconstruct_once(pix, mask_ratio)

        # Grey out masked patches on the original (0 in normalised space ~= mean grey).
        orig_patches = self.model.patchify(pix)
        m = mask.unsqueeze(-1)
        masked_patches = orig_patches * (1 - m)
        masked_view = self.model.unpatchify(masked_patches)

        return (
            self._denorm(masked_view).clamp(0.0, 1.0),
            self._denorm(recon_full_norm).clamp(0.0, 1.0),
            self._denorm(purified_norm).clamp(0.0, 1.0),
        )
