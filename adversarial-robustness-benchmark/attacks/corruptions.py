"""ImageNet-C-style common corruptions (Phase 1, Axis A — model-agnostic).

Implements the 15 standard ImageNet-C corruption types from Hendrycks &
Dietterich (ICLR 2019). All implementations are self-contained — pure
NumPy / SciPy / PIL — with no dependency on the original ``imagenet_c``
package (which requires ImageMagick / Wand and is painful to install on
Windows). Severity constants follow the canonical Hendrycks calibrations so
results are comparable to the published ImageNet-C numbers.

Two surfaces:

  * ``apply_corruption(arr_uint8_hwc, corruption_type, severity, rng)`` —
    pure function. Takes an HWC uint8 image, returns an HWC uint8 image.

  * ``CommonCorruptions(BaseAttack)`` — wraps the helper behind the standard
    ``apply(classifier, image_batch_0_1, labels)`` interface so the
    evaluation pipeline can drive it tensor-in / tensor-out.

Notes
-----
* ``motion_blur``, ``snow``, ``frost`` and ``fog`` in the canonical
  implementation rely on Wand/ImageMagick or pre-baked texture assets. The
  versions here approximate them in pure NumPy (directional convolution for
  motion blur, plasma fractal for fog, Gaussian-filtered noise for
  frost/snow). Numbers will be within the same order of magnitude as the
  reference but not bit-exact. The deviation is documented in the manifest.
* Corruption order follows the canonical 4-group taxonomy:
  noise / blur / weather / digital.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import zoom as scizoom

from .base import BaseAttack


CORRUPTION_TYPES: tuple[str, ...] = (
    # noise
    "gaussian_noise", "shot_noise", "impulse_noise",
    # blur
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    # weather
    "snow", "frost", "fog", "brightness",
    # digital
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def _disk_kernel(radius: int, alias_blur: float) -> np.ndarray:
    """Anti-aliased disk kernel used by defocus_blur."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((x ** 2 + y ** 2) <= radius ** 2).astype(np.float32)
    kernel = gaussian_filter(kernel, sigma=alias_blur)
    s = kernel.sum()
    return kernel / (s if s > 0 else 1.0)


def _convolve_per_channel(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution applied independently to each colour channel."""
    from scipy.ndimage import convolve as ndconvolve
    out = np.empty_like(x, dtype=np.float32)
    for d in range(x.shape[-1]):
        out[..., d] = ndconvolve(x[..., d].astype(np.float32), kernel, mode="reflect")
    return out


def gaussian_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = [.08, .12, .18, .26, .38][severity - 1]
    out = x.astype(np.float32) / 255.0 + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0) * 255.0


def shot_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    rate = [60, 25, 12, 5, 3][severity - 1]
    out = rng.poisson(x.astype(np.float32) / 255.0 * rate) / rate
    return np.clip(out, 0.0, 1.0) * 255.0


def impulse_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Salt-and-pepper noise — replaces a fraction of pixels with 0 or 255."""
    amount = [.03, .06, .09, .17, .27][severity - 1]
    out = x.astype(np.float32) / 255.0
    noise = rng.random(out.shape[:2])
    salt_mask = noise < amount / 2
    pepper_mask = (noise >= amount / 2) & (noise < amount)
    out[salt_mask] = 1.0
    out[pepper_mask] = 0.0
    return np.clip(out, 0.0, 1.0) * 255.0


def defocus_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    radius, alias_blur = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    kernel = _disk_kernel(radius, alias_blur)
    return np.clip(_convolve_per_channel(x, kernel), 0.0, 255.0)


def glass_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma, max_delta, iters = [
        (0.7, 1, 2), (0.9, 2, 1), (1.0, 2, 3), (1.1, 3, 2), (1.5, 4, 2)
    ][severity - 1]
    # Initial Gaussian blur over [0,1] image, then local pixel shuffling.
    x = np.uint8(gaussian_filter(x.astype(np.float32) / 255.0, sigma=(sigma, sigma, 0)) * 255)
    H, W = x.shape[:2]
    # Vectorised shuffle: pre-draw all (dx, dy) offsets, walk the grid in
    # row-major order, swap each pixel with the offset target. Faster than
    # the canonical nested Python loop but produces visually equivalent noise.
    for _ in range(iters):
        dxs = rng.integers(-max_delta, max_delta + 1, size=(H - 2 * max_delta, W - 2 * max_delta))
        dys = rng.integers(-max_delta, max_delta + 1, size=(H - 2 * max_delta, W - 2 * max_delta))
        for hi in range(H - max_delta - 1, max_delta, -1):
            for wi in range(W - max_delta - 1, max_delta, -1):
                dx = int(dxs[hi - max_delta, wi - max_delta])
                dy = int(dys[hi - max_delta, wi - max_delta])
                h2, w2 = hi + dy, wi + dx
                tmp = x[hi, wi].copy()
                x[hi, wi] = x[h2, w2]
                x[h2, w2] = tmp
    out = gaussian_filter(x.astype(np.float32) / 255.0, sigma=(sigma, sigma, 0)) * 255.0
    return np.clip(out, 0.0, 255.0)


def motion_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Directional blur — approximation of the canonical Wand-based version.

    Builds a line kernel at a random angle and convolves with it. Length /
    softness ramp per severity is calibrated to roughly match the canonical
    PSF widths.
    """
    radius, sigma = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    angle = rng.uniform(-45.0, 45.0)
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    rad = np.deg2rad(angle)
    cx = cy = size // 2
    for r in range(-radius, radius + 1):
        xo = int(round(cx + r * np.cos(rad)))
        yo = int(round(cy + r * np.sin(rad)))
        if 0 <= xo < size and 0 <= yo < size:
            kernel[yo, xo] = 1.0
    kernel = gaussian_filter(kernel, sigma=max(sigma * 0.25, 0.5))
    s = kernel.sum()
    kernel = kernel / (s if s > 0 else 1.0)
    return np.clip(_convolve_per_channel(x, kernel), 0.0, 255.0)


def zoom_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factors = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]
    x_arr = (x.astype(np.float32) / 255.0)
    H, W = x.shape[:2]
    acc = x_arr.copy()
    for zf in factors:
        zoomed = scizoom(x_arr, (zf, zf, 1), order=1)
        Hz, Wz = zoomed.shape[:2]
        sy, sx = (Hz - H) // 2, (Wz - W) // 2
        acc += zoomed[sy:sy + H, sx:sx + W]
    out = acc / (len(factors) + 1)
    return np.clip(out, 0.0, 1.0) * 255.0


def _plasma_fractal(map_size: int, wibbledecay: float, rng: np.random.Generator) -> np.ndarray:
    """Plasma-fractal cloud texture — power-of-2 square, returned in [0,1]."""
    s = 1
    while s < map_size:
        s *= 2
    arr = np.zeros((s, s), dtype=np.float32)
    arr[0, 0] = 0.0
    size = s
    wibble = 100.0
    while size > 1:
        half = size // 2
        # Diamond step
        for y in range(0, s, size):
            for xx in range(0, s, size):
                avg = (arr[y, xx] + arr[(y + size) % s, xx]
                       + arr[y, (xx + size) % s] + arr[(y + size) % s, (xx + size) % s]) / 4.0
                arr[y + half, xx + half] = avg + rng.uniform(-wibble, wibble)
        # Square step
        for y in range(0, s, size):
            for xx in range(0, s, size):
                avg_top = (arr[y, xx] + arr[y, (xx + size) % s]
                           + arr[(y - half) % s, (xx + half) % s] + arr[y + half, (xx + half) % s]) / 4.0
                avg_left = (arr[y, xx] + arr[(y + size) % s, xx]
                            + arr[(y + half) % s, (xx - half) % s] + arr[(y + half) % s, xx + half]) / 4.0
                arr[y, xx + half] = avg_top + rng.uniform(-wibble, wibble)
                arr[y + half, xx] = avg_left + rng.uniform(-wibble, wibble)
        wibble /= wibbledecay
        size = half
    arr -= arr.min()
    return arr / (arr.max() + 1e-9)


def snow(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic snow — Gaussian-noise streaks composited additively.

    Approximation of the canonical (Wand motion-blur + composite) version.
    """
    loc, scale, zoom_factor, thresh, blur_radius, _, layer_amount = [
        (0.10, 0.30, 3.0, 0.50, 10, 4, 0.80),
        (0.20, 0.30, 2.0, 0.50, 12, 4, 0.70),
        (0.55, 0.30, 4.0, 0.90, 12, 8, 0.70),
        (0.55, 0.30, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.30, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]
    H, W = x.shape[:2]
    snow_layer = rng.normal(loc=loc, scale=scale, size=(H, W)).astype(np.float32)
    snow_layer = scizoom(snow_layer, zoom_factor, order=1)
    snow_layer = snow_layer[:H, :W]
    snow_layer[snow_layer < thresh] = 0
    # Streak the snow with a vertical-ish motion-blur approximation.
    kernel_len = max(3, blur_radius)
    kernel = np.zeros((kernel_len, kernel_len), dtype=np.float32)
    for i in range(kernel_len):
        kernel[i, kernel_len // 2] = 1.0
    kernel = gaussian_filter(kernel, sigma=0.5)
    kernel /= kernel.sum()
    from scipy.ndimage import convolve as ndconvolve
    snow_layer = ndconvolve(snow_layer, kernel, mode="reflect")
    snow_layer = np.clip(snow_layer, 0.0, 1.0)
    x_arr = x.astype(np.float32) / 255.0
    grayscale = np.maximum(x_arr, np.expand_dims(snow_layer * layer_amount, -1).repeat(3, axis=2))
    out = grayscale + np.expand_dims(snow_layer, -1) * 0.5
    return np.clip(out * 255.0, 0.0, 255.0)


def frost(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic frost — Gaussian-blurred white noise composited onto the image.

    The canonical version blends pre-baked frost texture photos. We
    approximate with a low-frequency luminance field biased to white;
    perceptually similar (image gets a frosted-glass overlay) though not
    bit-equivalent.
    """
    img_amount, frost_amount = [
        (1.00, 0.40), (0.80, 0.60), (0.70, 0.70), (0.65, 0.70), (0.60, 0.75),
    ][severity - 1]
    H, W = x.shape[:2]
    noise = rng.random((H, W)).astype(np.float32)
    layer = gaussian_filter(noise, sigma=4.0)
    layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-9)
    layer = np.clip(layer * 0.6 + 0.4, 0.0, 1.0)  # bias toward bright/white
    rgb = np.repeat(layer[:, :, None], 3, axis=2) * 255.0
    out = img_amount * x.astype(np.float32) + frost_amount * rgb
    return np.clip(out, 0.0, 255.0)


def fog(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    intensity, wibbledecay = [(1.5, 2.0), (2.0, 2.0), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    H, W = x.shape[:2]
    max_side = max(H, W)
    fog_layer = _plasma_fractal(max_side, wibbledecay, rng)[:H, :W]
    fog_layer = np.repeat(fog_layer[:, :, None], 3, axis=2) * intensity
    x_arr = x.astype(np.float32) / 255.0
    out = x_arr + fog_layer
    # Re-scale so the brightest pixel matches the original brightest pixel —
    # avoids clipping to all-white.
    scale = max(x_arr.max(), 0.1) / (out.max() + 1e-9)
    return np.clip(out * scale * 255.0, 0.0, 255.0)


def brightness(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    delta = [.10, .20, .30, .40, .50][severity - 1]
    pil = Image.fromarray(x).convert("HSV")
    h, s, v = pil.split()
    v_arr = np.asarray(v).astype(np.float32) + delta * 255.0
    v_arr = np.clip(v_arr, 0.0, 255.0).astype(np.uint8)
    out = Image.merge("HSV", (h, s, Image.fromarray(v_arr, mode="L"))).convert("RGB")
    return np.asarray(out).astype(np.float32)


def contrast(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Lower contrast — scale pixels toward per-channel mean.

    Severity ordering matches Hendrycks: severity 5 = the lowest contrast.
    """
    factor = [.4, .3, .2, .1, .05][severity - 1]
    x_arr = x.astype(np.float32) / 255.0
    means = x_arr.mean(axis=(0, 1), keepdims=True)
    out = (x_arr - means) * factor + means
    return np.clip(out, 0.0, 1.0) * 255.0


def elastic_transform(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    alpha, sigma, _ = [
        (244 * 0.31, 244 * 0.06, 244 * 0.010),
        (244 * 0.31, 244 * 0.07, 244 * 0.005),
        (244 * 0.31, 244 * 0.08, 244 * 0.005),
        (244 * 0.31, 244 * 0.10, 244 * 0.0025),
        (244 * 0.31, 244 * 0.12, 244 * 0.003),
    ][severity - 1]
    H, W = x.shape[:2]
    dx = (gaussian_filter(rng.uniform(-1, 1, size=(H, W)).astype(np.float32),
                          sigma=sigma, mode="reflect") * alpha)
    dy = (gaussian_filter(rng.uniform(-1, 1, size=(H, W)).astype(np.float32),
                          sigma=sigma, mode="reflect") * alpha)
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([y_grid + dy, x_grid + dx], axis=0)
    out = np.empty_like(x, dtype=np.float32)
    for d in range(x.shape[-1]):
        out[..., d] = map_coordinates(x[..., d].astype(np.float32), coords, order=1, mode="reflect")
    return np.clip(out, 0.0, 255.0)


def pixelate(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    H, W = x.shape[:2]
    pil = Image.fromarray(x)
    tiny = pil.resize((max(1, int(W * factor)), max(1, int(H * factor))), Image.BOX)
    up = tiny.resize((W, H), Image.BOX)
    return np.asarray(up).astype(np.float32)


def jpeg_compression(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    quality = [25, 18, 15, 10, 7][severity - 1]
    pil = Image.fromarray(x)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.asarray(decoded).astype(np.float32)


_DISPATCH = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(
    image: np.ndarray | Image.Image,
    corruption_type: str,
    severity: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply ``corruption_type`` at ``severity`` to ``image``.

    Accepts a PIL.Image or an HWC uint8 numpy array. Returns an HWC uint8
    numpy array of the same H, W.
    """
    if corruption_type not in _DISPATCH:
        raise ValueError(
            f"unknown corruption_type {corruption_type!r}. "
            f"Must be one of {CORRUPTION_TYPES}."
        )
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in [1, 5], got {severity}")
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = image
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {arr.shape}")
    if rng is None:
        rng = np.random.default_rng()
    out = _DISPATCH[corruption_type](arr, severity, rng)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


class CommonCorruptions(BaseAttack):
    """BaseAttack-compliant wrapper around ``apply_corruption``.

    For dataset pre-generation prefer calling ``apply_corruption`` directly on
    a PIL image. This tensor path exists so the evaluation pipeline can call
    all attacks through the same interface.
    """

    name = "corruptions"

    def __init__(self, corruption_type: str, severity: int = 3, seed: int | None = None):
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(
                f"unknown corruption_type {corruption_type!r}. "
                f"Must be one of {CORRUPTION_TYPES}."
            )
        if not 1 <= severity <= 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")
        self.corruption_type = corruption_type
        self.severity = int(severity)
        self.seed = seed

    def apply(self, classifier, image_batch_0_1, labels):
        import torch
        rng = np.random.default_rng(self.seed)
        outs = []
        for i in range(image_batch_0_1.shape[0]):
            arr_chw = image_batch_0_1[i].detach().cpu().clamp(0, 1).numpy()
            arr_hwc = (arr_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            corrupted = apply_corruption(arr_hwc, self.corruption_type, self.severity, rng)
            outs.append(torch.from_numpy(corrupted.transpose(2, 0, 1).astype(np.float32) / 255.0))
        return torch.stack(outs, dim=0).to(image_batch_0_1.device).type_as(image_batch_0_1)


__all__ = [
    "CORRUPTION_TYPES",
    "apply_corruption",
    "CommonCorruptions",
]
