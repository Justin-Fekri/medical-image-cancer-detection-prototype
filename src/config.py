"""Central configuration for the cancer detection prototype.

Every module imports its settings from here so that the training script,
the evaluation script and the Streamlit app all agree on image size,
normalization statistics and the location of the checkpoint.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Every image is standardized to 224 x 224 RGB before it reaches the network.
IMAGE_SIZE = 224
NUM_CHANNELS = 3

# ImageNet statistics, because the EfficientNet-B0 backbone is pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Binary problem: index 0 = benign, index 1 = malignant.
CLASS_NAMES = ("benign", "malignant")


@dataclass
class TrainConfig:
    """Hyperparameters for a training run."""

    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_path: Path = PROJECT_ROOT / "models" / "efficientnet_b0_cancer.pt"

    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    # Freeze the pretrained feature extractor for the first N epochs so the
    # randomly initialized head does not destroy the pretrained weights.
    warmup_frozen_epochs: int = 3

    num_workers: int = 4
    seed: int = 42

    # Default operating point. The Streamlit app lets a reviewer move it.
    decision_threshold: float = 0.5

    # Early stopping on validation AUROC.
    patience: int = 5

    # Linear-probe mode: never unfreeze the backbone. Backpropagating through
    # EfficientNet-B0 is roughly 4x the cost of a head-only step on CPU, so
    # this is the practical setting for a machine without a GPU.
    freeze_backbone: bool = False

    augmentation: dict = field(
        default_factory=lambda: {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "rotation_degrees": 15,
            "color_jitter": 0.1,
        }
    )
