"""Run binary EfficientNet-B0 inference and attach Grad-CAM overlays."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image
from torch import nn

from .config import DEFAULT_THRESHOLD, NEGATIVE_LABEL, POSITIVE_LABEL
from .gradcam import GradCAM, colorize_heatmap, overlay_heatmap, upsample_heatmap
from .model import last_feature_layer
from .preprocess import preprocess_for_model


@dataclass(frozen=True)
class ThresholdDecision:
    probability: float
    threshold: float
    predicted_positive: bool
    label: str


@dataclass(frozen=True)
class InferenceResult:
    logit: float
    probability: float
    preview_rgb: np.ndarray
    heatmap: np.ndarray
    heatmap_color: np.ndarray
    overlay: np.ndarray
    original_size: tuple[int, int]


def apply_threshold(probability: float, threshold: float = DEFAULT_THRESHOLD) -> ThresholdDecision:
    """Map a sigmoid probability to a reviewer flag using an adjustable cutoff."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be in [0, 1].")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")
    positive = probability >= threshold
    return ThresholdDecision(
        probability=float(probability),
        threshold=float(threshold),
        predicted_positive=positive,
        label=POSITIVE_LABEL if positive else NEGATIVE_LABEL,
    )


def _to_device(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.to(device)
    return tensor.to(device)


def run_inference(
    model: nn.Module,
    source: str | Path | bytes | BinaryIO | Image.Image,
    device: str | torch.device | None = None,
) -> InferenceResult:
    """Preprocess to 224x224 RGB, score with sigmoid, and build a Grad-CAM overlay."""
    resolved = torch.device(device) if device else torch.device("cpu")
    tensor, preview, original = preprocess_for_model(source)
    tensor = _to_device(model, tensor, resolved)
    model.eval()

    with GradCAM(model, last_feature_layer(model)) as cam:
        logit = model(tensor)[0, 0]
        heatmap = cam.generate(logit)

    probability = torch.sigmoid(logit.detach()).item()
    color = colorize_heatmap(upsample_heatmap(heatmap, preview.shape[0], preview.shape[1]))
    overlay = overlay_heatmap(preview, heatmap)
    return InferenceResult(
        logit=float(logit.detach().cpu()),
        probability=float(probability),
        preview_rgb=preview,
        heatmap=heatmap,
        heatmap_color=color,
        overlay=overlay,
        original_size=original.size,
    )


def load_image_bytes(data: bytes) -> Image.Image:
    """Helper for Streamlit uploads."""
    from .preprocess import load_rgb_image

    return load_rgb_image(BytesIO(data))
