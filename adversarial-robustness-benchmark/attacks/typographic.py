"""Typographic overlay attack (Phase 1, Axis C — semantic / model-agnostic).

Render misleading class-name text onto an otherwise-clean image. The
perturbation is clearly visible (a sticker-like rectangle with text inside),
but physically plausible — a printed label, a sign, a watermark. Because the
attack does not depend on a target model, one corrupted dataset is generated
once and reused across all benchmarked models (project brief Section 3.3).

The paper hook (brief Section 3.6, hook #1): pure vision classifiers latch
onto the rendered text token and predict the wrong class, while a
language-grounded model (CLIP) matches against text descriptions of all
classes and resists the attack better. That gap is the central finding.

This module exposes two surfaces:

  * ``render_typographic(pil_image, text, ...)`` — the actual implementation.
    PIL-in, PIL-out, native resolution preserved so each classifier's
    preprocess pipeline can resize/crop it identically to a clean image.

  * ``TypographicOverlay(BaseAttack)`` — wraps the helper behind the standard
    ``apply(classifier, image_batch_0_1, labels)`` interface so the evaluation
    pipeline can drive it tensor-in / tensor-out.

Styling defaults reproduce the classic Goh-et-al. (OpenAI, 2021) typographic
attack: white rectangular sticker, bold black text, near the top of the image.
"""

from __future__ import annotations

import os
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .base import BaseAttack

# Font lookup order: prefer a bold sans-serif that's always installed. We try
# Windows system fonts first, then matplotlib's bundled DejaVuSans-Bold (always
# present because matplotlib is in requirements.txt), then PIL's default
# bitmap font as a last resort.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _mpl_dejavu_bold() -> str | None:
    """Return matplotlib's bundled DejaVuSans-Bold path, or None if absent."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return None
    import matplotlib as mpl
    path = os.path.join(
        os.path.dirname(mpl.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"
    )
    return path if os.path.exists(path) else None


def _find_font(size: int) -> ImageFont.ImageFont:
    """Return an ImageFont at the requested pixel size, or the bitmap default."""
    candidates = list(_FONT_CANDIDATES)
    mpl_font = _mpl_dejavu_bold()
    if mpl_font:
        candidates.append(mpl_font)
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Return (w, h, x_offset, y_offset) for the rendered text bounding box."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, -left, -top


def render_typographic(
    image: Image.Image,
    text: str,
    font_size_frac: float = 0.12,
    position: str = "top",
    padding_frac: float = 0.04,
    opacity: float = 1.0,
    bg_color: tuple = (255, 255, 255),
    fg_color: tuple = (0, 0, 0),
) -> Image.Image:
    """Render ``text`` onto ``image`` as a sticker, return a new PIL.Image.

    The sticker is a filled rectangle (``bg_color``) with the text drawn on
    top in ``fg_color``. Sizes are fractions of ``min(W, H)`` so the overlay
    scales with the image. The result is RGB.

      * ``font_size_frac``  — font size as a fraction of min(W, H).
      * ``position``        — 'top' | 'center' | 'bottom'.
      * ``padding_frac``    — padding inside the sticker, fraction of min(W,H).
      * ``opacity``         — sticker alpha in [0, 1]; 1.0 is fully opaque.
    """
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")
    if position not in {"top", "center", "bottom"}:
        raise ValueError(f"position must be one of top/center/bottom, got {position!r}")

    base = image.convert("RGB")
    W, H = base.size
    side = min(W, H)
    font_px = max(8, int(round(font_size_frac * side)))
    pad_px = max(2, int(round(padding_frac * side)))

    font = _find_font(font_px)

    # Measure on a scratch canvas — textbbox needs a draw context.
    scratch = Image.new("RGB", (1, 1))
    sd = ImageDraw.Draw(scratch)
    tw, th, tx_off, ty_off = _text_bbox(sd, text, font)

    box_w = min(W, tw + 2 * pad_px)
    box_h = th + 2 * pad_px

    # Center horizontally; vertical position chosen by `position`. A small
    # margin from the image edge keeps the sticker fully on-canvas.
    edge_margin = max(2, int(round(0.02 * side)))
    box_x = (W - box_w) // 2
    if position == "top":
        box_y = edge_margin
    elif position == "bottom":
        box_y = H - box_h - edge_margin
    else:  # center
        box_y = (H - box_h) // 2

    # Draw onto an RGBA overlay so opacity < 1 actually blends. Compositing
    # back to RGB at the end (the manifest format is JPEG → RGB anyway).
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    alpha = int(round(255 * opacity))
    od.rectangle(
        [box_x, box_y, box_x + box_w - 1, box_y + box_h - 1],
        fill=(bg_color[0], bg_color[1], bg_color[2], alpha),
    )
    text_x = box_x + pad_px + tx_off + (box_w - 2 * pad_px - tw) // 2
    text_y = box_y + pad_px + ty_off
    od.text((text_x, text_y), text, font=font, fill=(fg_color[0], fg_color[1], fg_color[2], alpha))

    composed = Image.alpha_composite(base.convert("RGBA"), overlay)
    return composed.convert("RGB")


class TypographicOverlay(BaseAttack):
    """BaseAttack-compliant wrapper around ``render_typographic``.

    Drives the renderer over a tensor batch by round-tripping through PIL.
    For dataset generation prefer calling ``render_typographic`` directly on
    the native-resolution PIL image; this tensor path exists so the evaluation
    pipeline can call all attacks through a single interface.
    """

    name = "typographic"

    def __init__(
        self,
        targets: Sequence[str] | None = None,
        font_size_frac: float = 0.12,
        position: str = "top",
        padding_frac: float = 0.04,
        opacity: float = 1.0,
    ):
        self.targets = list(targets) if targets is not None else None
        self.font_size_frac = float(font_size_frac)
        self.position = str(position)
        self.padding_frac = float(padding_frac)
        self.opacity = float(opacity)

    def apply(self, classifier, image_batch_0_1, labels):
        import torch
        from torchvision.transforms.functional import to_pil_image, to_tensor

        if self.targets is None:
            raise ValueError(
                "TypographicOverlay.apply needs ``targets`` set at construction "
                "(one target string per batch element)."
            )
        if len(self.targets) != image_batch_0_1.shape[0]:
            raise ValueError(
                f"len(targets)={len(self.targets)} != batch={image_batch_0_1.shape[0]}"
            )

        outs = []
        for i in range(image_batch_0_1.shape[0]):
            pil = to_pil_image(image_batch_0_1[i].detach().cpu().clamp(0, 1))
            adv_pil = render_typographic(
                pil,
                self.targets[i],
                font_size_frac=self.font_size_frac,
                position=self.position,
                padding_frac=self.padding_frac,
                opacity=self.opacity,
            )
            outs.append(to_tensor(adv_pil))
        return torch.stack(outs, dim=0).to(image_batch_0_1.device).type_as(image_batch_0_1)
