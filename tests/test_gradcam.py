import numpy as np
import torch
from PIL import Image

from meddetect.gradcam import GradCAM, overlay_heatmap
from meddetect.inference import run_inference
from meddetect.model import build_model, last_feature_layer
from meddetect.synthetic import render_negative, render_positive


def test_gradcam_heatmap_is_normalized() -> None:
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    with GradCAM(model, last_feature_layer(model)) as cam:
        logit = model(x)[0, 0]
        heatmap = cam.generate(logit)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_overlay_matches_image_shape() -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap = np.linspace(0, 1, 7 * 7, dtype=np.float32).reshape(7, 7)
    overlay = overlay_heatmap(image, heatmap)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_inference_on_synthetic_images() -> None:
    model = build_model(pretrained=False)
    model.eval()
    for factory in (render_negative, render_positive):
        result = run_inference(model, factory(size=180, seed=3))
        assert 0.0 <= result.probability <= 1.0
        assert result.preview_rgb.shape == (224, 224, 3)
        assert result.overlay.shape == (224, 224, 3)
        assert result.heatmap_color.shape == (224, 224, 3)


def test_inference_accepts_grayscale_pil() -> None:
    model = build_model(pretrained=False)
    model.eval()
    gray = Image.fromarray(np.full((100, 120), 90, dtype=np.uint8), mode="L")
    result = run_inference(model, gray)
    assert result.preview_rgb.shape[2] == 3
