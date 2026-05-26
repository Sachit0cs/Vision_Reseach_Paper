"""Surrogate-based transfer attack harness (Phase B).

PGD adversarial images crafted on a single white-box surrogate, then reused
verbatim on every benchmarked target model. The harness only handles
*generation* — evaluation is a separate step that simply feeds the saved
tensors to each target model's ``logits``.

Why this lives apart from ``attacks/gradient.py``:
  * the gradient attacks are model-specific (regenerated per target);
  * the transfer set is generated once on a fixed surrogate and pinned to disk
    so cross-architecture results are bit-reproducible.

The harness preprocesses every clean image through the *surrogate's* resize +
crop pipeline (NOT normalization) and emits a shared ``[N, 3, 224, 224]``
float32 tensor in [0, 1] pixel space. Targets are expected to accept that
shared tensor directly via ``classifier.logits`` (which then applies its own
mean/std internally — see ``BaseClassifier`` contract).
"""

from __future__ import annotations

import time
from typing import Iterable

import torch

from .gradient import PGD


class TransferHarness:
    """Crafts adversarial images on a surrogate; saves clean + adv + labels."""

    def __init__(
        self,
        surrogate,
        epsilon: float,
        step_size: float,
        num_steps: int,
        seed: int,
        random_start: bool = True,
    ):
        self.surrogate = surrogate
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.random_start = bool(random_start)
        # One PGD instance, reused across mini-batches. Seed gives random-start
        # noise that is reproducible across runs.
        self._pgd = PGD(
            epsilon=self.epsilon,
            step_size=self.step_size,
            num_steps=self.num_steps,
            random_start=self.random_start,
            seed=self.seed,
        )

    def preprocess_clean(self, pil_images: Iterable) -> torch.Tensor:
        """Run the surrogate's preprocess on each PIL image, stack to a batch.

        Returns a CPU float32 tensor in [0, 1]. The caller decides when to
        move it to the device.
        """
        tensors = [self.surrogate.preprocess(im) for im in pil_images]
        return torch.stack(tensors, dim=0).float()

    def craft(
        self,
        clean_batch_0_1: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32,
        progress: bool = True,
    ) -> torch.Tensor:
        """PGD against the surrogate, mini-batched. Returns adv tensor on CPU.

        Inputs may live on CPU; each mini-batch is moved to the surrogate
        device, attacked, and pulled back to CPU before concatenation so peak
        GPU memory stays bounded by ``batch_size``.
        """
        device = self.surrogate.device
        n = clean_batch_0_1.shape[0]
        out_chunks: list[torch.Tensor] = []
        t0 = time.time()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = clean_batch_0_1[start:end].to(device)
            y = labels[start:end].to(device)
            adv = self._pgd.apply(self.surrogate, x, y)
            out_chunks.append(adv.detach().cpu())
            if progress:
                done = end
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(
                    f"  PGD batch {done:>5d}/{n}  "
                    f"elapsed={elapsed/60:5.1f} min  "
                    f"eta={eta/60:5.1f} min  "
                    f"({rate:.2f} img/s)",
                    flush=True,
                )
        return torch.cat(out_chunks, dim=0)

    def verify_linf(self, clean_batch_0_1: torch.Tensor, adv_batch_0_1: torch.Tensor) -> float:
        """Return max |adv - clean| and raise if it exceeds epsilon + 1e-6."""
        diff = (adv_batch_0_1 - clean_batch_0_1).abs()
        max_dev = float(diff.max())
        if max_dev > self.epsilon + 1e-6:
            raise RuntimeError(
                f"L-infinity constraint violated: max|adv-clean|={max_dev:.6f} "
                f"> epsilon+1e-6={self.epsilon + 1e-6:.6f}"
            )
        return max_dev
