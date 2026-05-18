"""BaseAttack interface (project brief, Section 5.4 — STUB).

Every attack implements::

    apply(classifier, image_batch_0_1, labels) -> adversarial_batch_0_1

All tensors live in [0, 1] pixel space. L-infinity epsilon is the primary
threat model and is bounded in that space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Common interface for every attack."""

    name: str

    @abstractmethod
    def apply(self, classifier, image_batch_0_1, labels):
        """Return an adversarial batch in [0, 1], same shape as the input."""
        raise NotImplementedError
