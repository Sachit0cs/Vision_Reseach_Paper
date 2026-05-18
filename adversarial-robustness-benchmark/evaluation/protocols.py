"""White-box vs transfer protocol logic (Phase 2 — STUB).

Encodes Section 3.3: gradient attacks (FGSM/PGD/AutoAttack/Square) are
regenerated per model under the white-box protocol; the transfer protocol
crafts once on a fixed surrogate; semantic attacks (patch/typographic/
corruptions) reuse the shared model-agnostic poisoned set. White-box and
transfer results are reported as SEPARATE tables — never mixed.
"""

from __future__ import annotations

WHITE_BOX = "white_box"
TRANSFER = "transfer"
SHARED = "shared"


def protocol_for(attack_name: str) -> str:
    """Return the protocol an attack belongs to."""
    raise NotImplementedError("Phase 2: implement protocol routing.")
