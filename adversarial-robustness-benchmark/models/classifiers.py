"""The 7 benchmarked architecture wrappers + factory (Phase 0).

Design rule (project brief, Section 5.3): ``preprocess`` produces tensors in
[0, 1] pixel space and normalization is a SEPARATE step inside ``logits``, so
adversarial attacks can bound epsilon in pixel space and still differentiate
through the model.

  * TorchVisionClassifier — ResNet-50, VGG-16, ConvNeXt-Tiny, ViT-B/16,
    Swin-Tiny, EfficientNet-B0 (torchvision, DEFAULT weights).
  * ClipZeroShotClassifier — CLIP ViT-B/16 used as a zero-shot classifier.
  * build_classifier(name, device) — factory keyed by the short names in
    config.yaml.

Heavy libraries (torchvision, transformers) are imported lazily inside the
constructors so this module imports cleanly even where they are absent.
"""

from __future__ import annotations

import torch

from .base import BaseClassifier

# torchvision short key -> (model builder name, weights enum name).
_TV_SPECS = {
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "vgg16": ("vgg16", "VGG16_Weights"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
    "swin_t": ("swin_t", "Swin_T_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}

_CLIP_KEY = "clip_vit_b16"
_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"


def _scalar(v):
    """torchvision presets store sizes as 1-element lists; unwrap to a scalar."""
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return v[0]
    return v


class TorchVisionClassifier(BaseClassifier):
    """Wrapper for an ImageNet-pretrained torchvision CNN / Transformer."""

    def __init__(self, name: str, device: str = "cpu"):
        if name not in _TV_SPECS:
            raise ValueError(f"Unknown torchvision model key: {name}")
        from torchvision import models as tvm
        from torchvision import transforms as T

        builder_name, weights_enum = _TV_SPECS[name]
        weights = getattr(tvm, weights_enum).DEFAULT

        self.name = name
        self.device = device
        self.categories = list(weights.meta["categories"])
        self.model = getattr(tvm, builder_name)(weights=weights).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Split torchvision's bundled transform: keep resize + crop + to-tensor
        # ([0, 1] space), pull mean/std out into a separate normalization step.
        tf = weights.transforms()
        self._preprocess = T.Compose(
            [
                T.Resize(_scalar(tf.resize_size), interpolation=tf.interpolation, antialias=True),
                T.CenterCrop(_scalar(tf.crop_size)),
                T.ToTensor(),  # PIL -> (3, H, W) float in [0, 1]
            ]
        )
        self._mean = torch.tensor(tf.mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(tf.std, device=device).view(1, 3, 1, 1)

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    def logits(self, batch_0_1):
        x = batch_0_1.to(self.device)
        x = (x - self._mean) / self._std
        return self.model(x)

    def predict(self, image):
        x = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            probs = self.logits(x).softmax(dim=1)
        conf, idx = probs.max(dim=1)
        return self.categories[int(idx)], float(conf)


class ClipZeroShotClassifier(BaseClassifier):
    """CLIP (ViT-B/16) used as a zero-shot ImageNet classifier.

    Classifies by matching image features against text features of each class
    name. Text features are precomputed once; ``logits`` stays differentiable
    w.r.t. the input so white-box attacks work.
    """

    name = _CLIP_KEY

    def __init__(self, device: str = "cpu", categories=None, prompt: str = "a photo of a {}."):
        if categories is None:
            raise ValueError("ClipZeroShotClassifier needs the list of class names.")
        from transformers import CLIPModel, CLIPProcessor
        from torchvision import transforms as T

        self.device = device
        self.categories = list(categories)
        self.model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        img_proc = processor.image_processor
        size = img_proc.crop_size["height"]
        self._preprocess = T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(size),
                T.ToTensor(),  # (3, H, W) float in [0, 1]
            ]
        )
        self._mean = torch.tensor(img_proc.image_mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(img_proc.image_std, device=device).view(1, 3, 1, 1)

        # Precompute L2-normalized text features for every class (once).
        clean = [c.split(",")[0].strip().lower() for c in self.categories]
        prompts = [prompt.format(c) for c in clean]
        with torch.no_grad():
            tokens = processor(text=prompts, return_tensors="pt", padding=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            text_features = self.model.get_text_features(**tokens)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features  # (num_classes, dim)
        self._logit_scale = self.model.logit_scale.exp()

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    def logits(self, batch_0_1):
        x = batch_0_1.to(self.device)
        x = (x - self._mean) / self._std
        image_features = self.model.get_image_features(pixel_values=x)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self._logit_scale * image_features @ self._text_features.t()

    def predict(self, image):
        x = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            probs = self.logits(x).softmax(dim=1)
        conf, idx = probs.max(dim=1)
        return self.categories[int(idx)], float(conf)


def _imagenet_categories() -> list[str]:
    """The 1000 index-aligned ImageNet class names (from torchvision metadata)."""
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.DEFAULT.meta["categories"])


def build_classifier(name: str, device: str = "cpu") -> BaseClassifier:
    """Factory: short model key -> BaseClassifier instance."""
    if name in _TV_SPECS:
        return TorchVisionClassifier(name, device)
    if name == _CLIP_KEY:
        return ClipZeroShotClassifier(device=device, categories=_imagenet_categories())
    known = sorted(_TV_SPECS) + [_CLIP_KEY]
    raise ValueError(f"Unknown model key '{name}'. Known keys: {known}")
