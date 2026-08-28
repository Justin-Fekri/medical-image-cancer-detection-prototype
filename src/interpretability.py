"""Grad-CAM saliency maps, implemented directly on top of torch hooks.

Grad-CAM weights each channel of the last convolutional feature map by the
gradient of the output logit with respect to that channel, then sums the
weighted channels. The result highlights the regions that pushed the score
up. It is an explanation aid for a human reviewer, not evidence that the
model reasoned correctly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import IMAGE_SIZE
from .model import CancerClassifier


class GradCAM:
    """Grad-CAM for a single-logit binary classifier.

    Usage::

        with GradCAM(model) as cam:
            heatmap = cam(input_tensor)
    """

    def __init__(self, model: CancerClassifier) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        layer = self.model.target_layer
        self._handles.append(layer.register_forward_hook(self._save_activations))
        self._handles.append(layer.register_full_backward_hook(self._save_gradients))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return a ``(IMAGE_SIZE, IMAGE_SIZE)`` heatmap normalized to ``[0, 1]``.

        ``input_tensor`` is a single preprocessed image of shape
        ``(1, 3, 224, 224)``.
        """
        if not self._handles:
            raise RuntimeError("Use GradCAM inside a `with` block so hooks are active.")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Gradients are required here even though callers are usually inside
        # an inference path, so enable grad explicitly.
        with torch.enable_grad():
            logit = self.model(input_tensor.requires_grad_(True))
            logit.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks captured nothing; check the target layer.")

        # One weight per channel: the spatial mean of that channel's gradient.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        peak = cam.max()
        # A flat map means no region stood out; return zeros rather than NaNs.
        return cam / peak if peak > 0 else np.zeros_like(cam)


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Turn a ``[0, 1]`` heatmap into an RGB uint8 array.

    Implements a blue -> green -> red ramp by hand so the project does not
    need matplotlib at inference time.
    """
    h = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 * h - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * h - 1.5), 0.0, 1.0)
    blue = np.clip(0.5 - 1.5 * h, 0.0, 1.0) + np.clip(1.0 - 3.0 * h, 0.0, 1.0) * 0.5
    rgb = np.stack([red, green, np.clip(blue, 0.0, 1.0)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a heatmap over the standardized version of the original image."""
    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    base_arr = np.asarray(base).astype(np.float32)
    heat_arr = apply_colormap(heatmap).astype(np.float32)

    # Weight the blend by intensity so quiet regions stay close to the original.
    weight = (alpha * heatmap)[..., None]
    blended = base_arr * (1 - weight) + heat_arr * weight
    return Image.fromarray(blended.astype(np.uint8))
