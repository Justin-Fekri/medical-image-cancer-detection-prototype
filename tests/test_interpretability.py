"""Grad-CAM produces a usable map even on an untrained network."""

import numpy as np
import torch
from PIL import Image

from src.config import IMAGE_SIZE
from src.interpretability import GradCAM, apply_colormap, overlay_heatmap
from src.model import CancerClassifier


def test_gradcam_returns_a_normalized_map():
    model = CancerClassifier(pretrained=False)
    with GradCAM(model) as cam:
        heatmap = cam(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    assert heatmap.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_hooks_are_removed_on_exit():
    model = CancerClassifier(pretrained=False)
    with GradCAM(model) as cam:
        pass
    assert cam._handles == []


def test_calling_outside_the_context_manager_raises():
    cam = GradCAM(CancerClassifier(pretrained=False))
    try:
        cam(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    except RuntimeError:
        return
    raise AssertionError("Expected a RuntimeError when hooks are not registered.")


def test_colormap_output_is_rgb_uint8():
    rgb = apply_colormap(np.linspace(0, 1, IMAGE_SIZE * IMAGE_SIZE).reshape(IMAGE_SIZE, -1))
    assert rgb.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert rgb.dtype == np.uint8


def test_overlay_matches_the_standardized_size():
    image = Image.new("L", (512, 300), color=90)
    heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    overlay = overlay_heatmap(image, heatmap)

    assert overlay.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert overlay.mode == "RGB"
