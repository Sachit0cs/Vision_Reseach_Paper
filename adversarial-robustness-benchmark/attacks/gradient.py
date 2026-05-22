"""White-box / black-box gradient attacks (Phase 1).

Four attacks, all L-infinity, all operating in [0, 1] pixel space:

  * FGSM   — single-step, weak baseline (implemented directly).
  * PGD    — multi-step, the standard workhorse (implemented directly).
  * AutoAttack — the gold-standard 4-attack ensemble (official `autoattack`).
  * Square — score-based black-box attack (`torchattacks`).

These are MODEL-SPECIFIC: the adversarial image depends on (image, model,
label), so under the white-box protocol they are regenerated against each
architecture. ``apply`` follows the BaseAttack interface:

    apply(classifier, image_batch_0_1, labels) -> adversarial_batch_0_1

Epsilon is a constructor parameter (not hard-coded) so the evaluation pipeline
can run a single value or sweep several.

Sanity checks the results must satisfy (project brief, Section 7):
  * PGD must measurably reduce a weak model's (VGG-16) accuracy;
  * AutoAttack robust accuracy <= PGD robust accuracy;
  * Square (black-box) must not dramatically beat PGD — if it does, the model's
    gradients are masked and its apparent robustness is an artifact.

`autoattack` and `torchattacks` are imported lazily inside ``apply`` so this
module imports even when they are absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttack

_EPS = 8 / 255
_STEP = 2 / 255


class _ClassifierModule(nn.Module):
    """Adapts a BaseClassifier into a plain nn.Module: [0,1] batch -> logits.

    External attack libraries (autoattack, torchattacks) expect an nn.Module
    with a ``forward``; our classifiers expose ``.logits`` instead. The
    classifier applies its own normalization internally, so the module sees
    pure [0, 1] input — exactly what the attack libraries assume.
    """

    def __init__(self, classifier):
        super().__init__()
        self._classifier = classifier

    def forward(self, x):
        return self._classifier.logits(x)


def _prepare(classifier, image_batch_0_1, labels):
    """Move a batch + labels onto the classifier's device with correct dtypes."""
    device = classifier.device
    images = image_batch_0_1.clone().detach().to(device).float()
    labels = torch.as_tensor(labels, device=device).long()
    return device, images, labels


class FGSM(BaseAttack):
    """Fast Gradient Sign Method — single-step L-infinity attack."""

    name = "fgsm"

    def __init__(self, epsilon: float = _EPS):
        self.epsilon = float(epsilon)

    def apply(self, classifier, image_batch_0_1, labels):
        _, images, labels = _prepare(classifier, image_batch_0_1, labels)

        # enable_grad guard: callers (e.g. evaluation loops) often wrap inference
        # in torch.no_grad(); without this, autograd.grad would fail.
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)
            loss = F.cross_entropy(classifier.logits(images), labels)
            grad = torch.autograd.grad(loss, images)[0]

        adv = images.detach() + self.epsilon * grad.sign()
        return torch.clamp(adv, 0.0, 1.0).detach()


class PGD(BaseAttack):
    """Projected Gradient Descent — multi-step L-infinity attack.

    The standard workhorse: take ``num_steps`` signed-gradient steps of size
    ``step_size``, projecting back into the epsilon-ball after each step.
    """

    name = "pgd"

    def __init__(
        self,
        epsilon: float = _EPS,
        step_size: float = _STEP,
        num_steps: int = 20,
        random_start: bool = True,
        seed: int | None = None,
    ):
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.random_start = bool(random_start)
        self.seed = seed

    def apply(self, classifier, image_batch_0_1, labels):
        device, images, labels = _prepare(classifier, image_batch_0_1, labels)

        if self.random_start:
            noise = torch.empty_like(images)
            if self.seed is not None:
                g = torch.Generator(device=device).manual_seed(self.seed)
                noise.uniform_(-self.epsilon, self.epsilon, generator=g)
            else:
                noise.uniform_(-self.epsilon, self.epsilon)
            adv = torch.clamp(images + noise, 0.0, 1.0)
        else:
            adv = images.clone()

        # enable_grad guard: callers (e.g. evaluation loops) often wrap inference
        # in torch.no_grad(); without this, autograd.grad would fail.
        with torch.enable_grad():
            for _ in range(self.num_steps):
                adv = adv.detach().requires_grad_(True)
                loss = F.cross_entropy(classifier.logits(adv), labels)
                grad = torch.autograd.grad(loss, adv)[0]

                adv = adv.detach() + self.step_size * grad.sign()
                # Project back into the L-infinity epsilon-ball, then into [0, 1].
                delta = torch.clamp(adv - images, -self.epsilon, self.epsilon)
                adv = torch.clamp(images + delta, 0.0, 1.0)

        return adv.detach()


class AutoAttackWrapper(BaseAttack):
    """AutoAttack — the parameter-free 4-attack ensemble (gold standard).

    Uses the official ``autoattack`` package (APGD-CE, APGD-T, FAB-T, Square).
    Non-negotiable for a credible robustness claim.
    """

    name = "autoattack"

    def __init__(
        self,
        epsilon: float = _EPS,
        norm: str = "Linf",
        version: str = "standard",
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.epsilon = float(epsilon)
        self.norm = norm
        self.version = version
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def apply(self, classifier, image_batch_0_1, labels):
        from autoattack import AutoAttack

        device, images, labels = _prepare(classifier, image_batch_0_1, labels)
        model = _ClassifierModule(classifier).eval()

        adversary = AutoAttack(
            model,
            norm=self.norm,
            eps=self.epsilon,
            version=self.version,
            device=device,
            seed=self.seed,
        )
        adv = adversary.run_standard_evaluation(images, labels, bs=self.batch_size)
        return adv.detach()


class SquareAttack(BaseAttack):
    """Square Attack — score-based black-box L-infinity attack.

    Queries only the model's outputs (no gradients). Used as the
    gradient-masking sanity check against the white-box attacks.
    """

    name = "square"

    def __init__(self, epsilon: float = _EPS, n_queries: int = 5000, seed: int = 42):
        self.epsilon = float(epsilon)
        self.n_queries = int(n_queries)
        self.seed = int(seed)

    def apply(self, classifier, image_batch_0_1, labels):
        import torchattacks

        _, images, labels = _prepare(classifier, image_batch_0_1, labels)
        model = _ClassifierModule(classifier).eval()

        attack = torchattacks.Square(
            model,
            norm="Linf",
            eps=self.epsilon,
            n_queries=self.n_queries,
            seed=self.seed,
            verbose=False,
        )
        return attack(images, labels).detach()


_GRADIENT_ATTACKS = {
    "fgsm": FGSM,
    "pgd": PGD,
    "autoattack": AutoAttackWrapper,
    "square": SquareAttack,
}


def build_attack(name: str, **kwargs) -> BaseAttack:
    """Factory: attack name -> attack instance. Extra kwargs go to the ctor."""
    if name not in _GRADIENT_ATTACKS:
        raise ValueError(
            f"Unknown gradient attack '{name}'. Known: {sorted(_GRADIENT_ATTACKS)}"
        )
    return _GRADIENT_ATTACKS[name](**kwargs)
