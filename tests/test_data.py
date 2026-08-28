"""The standardization contract: anything in, 1 x 3 x 224 x 224 out."""

import numpy as np
import torch
from PIL import Image

from src.config import IMAGE_SIZE
from src.data import build_eval_transform, preprocess_pil


def _random_image(mode: str, size: tuple[int, int]) -> Image.Image:
    channels = len(Image.new(mode, (1, 1)).getbands())
    array = np.random.randint(0, 255, (*size[::-1], channels), dtype=np.uint8)
    return Image.fromarray(array.squeeze(), mode=mode)


def test_preprocess_returns_expected_batch_shape():
    tensor = preprocess_pil(_random_image("RGB", (640, 480)))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_grayscale_is_expanded_to_three_channels():
    # Medical images are frequently single-channel; the pipeline must not choke.
    tensor = preprocess_pil(_random_image("L", (300, 300)))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_rgba_alpha_channel_is_dropped():
    tensor = preprocess_pil(_random_image("RGBA", (128, 256)))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_non_square_input_is_resized_not_cropped():
    # A very wide image still comes out square, so nothing is silently lost
    # to a center crop the reviewer never sees.
    tensor = preprocess_pil(_random_image("RGB", (1000, 100)))
    assert tensor.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE)


def test_normalization_shifts_the_distribution():
    # A mid-gray image should land near zero after ImageNet normalization.
    gray = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
    tensor = build_eval_transform()(gray)
    assert abs(float(tensor.mean())) < 0.15
