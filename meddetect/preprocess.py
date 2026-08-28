"""Standardize medical images to 224x224 RGB tensors."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


class ImageLoadError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


def rgba_to_rgb(image: Image.Image) -> Image.Image:
    """Composite RGBA onto black and return a 3-channel RGB image."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def load_rgb_image(source: str | Path | bytes | BinaryIO) -> Image.Image:
    """Load any common image format and convert it to RGB.

    Grayscale, palette, and RGBA inputs are all promoted to 3-channel RGB so
    EfficientNet-B0 always receives the layout it was designed for.
    """
    try:
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        elif isinstance(source, bytes):
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError("Could not decode the file as an image.") from exc
    return rgba_to_rgb(image)


def resize_square_rgb(image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """Resize to a square RGB image with bilinear resampling."""
    rgb = rgba_to_rgb(image)
    if rgb.size == (size, size):
        return rgb
    return rgb.resize((size, size), Image.Resampling.BILINEAR)


def to_uint8_array(image: Image.Image) -> np.ndarray:
    """Return a writable HWC uint8 RGB array."""
    array = np.array(rgba_to_rgb(image), dtype=np.uint8, copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageLoadError("Expected an RGB image array with 3 channels.")
    return array


def image_to_tensor(
    image: Image.Image,
    size: int = INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return a normalized CHW float tensor and the 224x224 uint8 preview."""
    resized = resize_square_rgb(image, size=size)
    preview = to_uint8_array(resized)
    tensor = torch.from_numpy(preview).float().permute(2, 0, 1) / 255.0
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor, preview


def preprocess_for_model(
    source: str | Path | bytes | BinaryIO | Image.Image,
    size: int = INPUT_SIZE,
) -> tuple[torch.Tensor, np.ndarray, Image.Image]:
    """Full inference preprocess: original RGB image, 224x224 preview, tensor."""
    original = rgba_to_rgb(source) if isinstance(source, Image.Image) else load_rgb_image(source)
    tensor, preview = image_to_tensor(original, size=size)
    return tensor.unsqueeze(0), preview, original
