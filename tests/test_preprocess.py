from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from meddetect.config import INPUT_SIZE
from meddetect.preprocess import (
    ImageLoadError,
    image_to_tensor,
    load_rgb_image,
    preprocess_for_model,
    resize_square_rgb,
)


def test_load_rgb_promotes_grayscale() -> None:
    gray = Image.fromarray(np.full((80, 50), 130, dtype=np.uint8), mode="L")
    path = Path("/tmp/meddetect_gray.png")
    gray.save(path)
    rgb = load_rgb_image(path)
    assert rgb.mode == "RGB"
    assert rgb.size == (50, 80)


def test_resize_is_224_square() -> None:
    image = Image.new("RGB", (640, 480), color=(12, 34, 56))
    resized = resize_square_rgb(image)
    assert resized.size == (INPUT_SIZE, INPUT_SIZE)
    assert resized.mode == "RGB"


def test_tensor_shape_and_normalization() -> None:
    image = Image.new("RGB", (300, 200), color=(255, 0, 0))
    tensor, preview = image_to_tensor(image)
    assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert preview.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert preview.dtype == np.uint8
    # Red channel should be above ImageNet mean after normalization.
    assert float(tensor[0].mean()) > 1.0
    assert float(tensor[1].mean()) < 0.0


def test_preprocess_for_model_adds_batch_dim() -> None:
    image = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))
    batch, preview, original = preprocess_for_model(image)
    assert batch.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert original.mode == "RGB"
    assert preview.shape[0] == INPUT_SIZE


def test_corrupt_bytes_raise() -> None:
    with pytest.raises(ImageLoadError):
        load_rgb_image(b"this is not an image")
