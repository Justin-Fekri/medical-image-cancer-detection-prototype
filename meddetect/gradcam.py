"""Grad-CAM heatmaps for reviewer oversight of EfficientNet-B0 decisions."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from torch import nn


class GradCAM:
    """Compute a class-activation heatmap from the last convolutional block."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def generate(self, logit: torch.Tensor) -> np.ndarray:
        """Backprop through a single logit and return a [0, 1] spatial heatmap."""
        if logit.numel() != 1:
            raise ValueError("Grad-CAM expects a scalar logit for one image.")

        self.model.zero_grad(set_to_none=True)
        logit.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        activations = self._activations[0]
        gradients = self._gradients[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = cam.cpu().numpy()
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam.astype(np.float32)


def upsample_heatmap(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear-upsample a 2D heatmap to the display resolution."""
    tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    scaled = torch.nn.functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return scaled.squeeze().cpu().numpy().astype(np.float32)


def colorize_heatmap(heatmap: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Map a [0, 1] heatmap to an RGB uint8 image."""
    cmap = colormaps[cmap_name]
    colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return (colored * 255.0).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colorized Grad-CAM map onto an RGB uint8 image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Overlay requires an HWC RGB image.")
    h, w = image.shape[:2]
    heat = upsample_heatmap(heatmap, h, w)
    color = colorize_heatmap(heat)
    blend = (1.0 - alpha) * image.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)
