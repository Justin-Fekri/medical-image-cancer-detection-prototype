"""Shared constants for the EfficientNet-B0 prototype."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "prototype_efficientnet_b0.pt"
SAMPLES_DIR = ROOT / "samples"

ARCH_NAME = "efficientnet_b0"
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_THRESHOLD = 0.50
POSITIVE_LABEL = "Above threshold (flagged for review)"
NEGATIVE_LABEL = "Below threshold (not flagged)"

DISCLAIMER = (
    "Educational and research-only prototype — not a diagnostic device. "
    "Do not use these scores for screening, diagnosis, or treatment decisions."
)
