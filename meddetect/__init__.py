"""Medical image cancer detection prototype (educational / research only)."""

from .config import DEFAULT_THRESHOLD, INPUT_SIZE
from .inference import InferenceResult, apply_threshold, run_inference
from .model import build_model, load_checkpoint
from .preprocess import image_to_tensor, load_rgb_image, preprocess_for_model

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_THRESHOLD",
    "INPUT_SIZE",
    "InferenceResult",
    "apply_threshold",
    "build_model",
    "image_to_tensor",
    "load_checkpoint",
    "load_rgb_image",
    "preprocess_for_model",
    "run_inference",
]
