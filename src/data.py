"""Dataset construction and the image standardization pipeline.

The directory layout expected under ``data/`` is the standard
``ImageFolder`` convention::

    data/
      train/benign/*.png
      train/malignant/*.png
      val/benign/*.png
      val/malignant/*.png
      test/...

Class index 0 is ``benign`` and class index 1 is ``malignant`` because
``ImageFolder`` sorts the class directories alphabetically, which matches
``config.CLASS_NAMES``.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, TrainConfig


class ConvertToRGB:
    """Force three channels, whatever the source mode.

    A module-level class rather than an inline lambda: DataLoader workers
    are spawned (not forked) on macOS and Windows, so every element of the
    transform pipeline has to be picklable. A lambda is not, and it fails
    only once ``num_workers > 0``.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_eval_transform() -> transforms.Compose:
    """Standardize a single image to 224 x 224 RGB, normalized.

    Used for validation, test and single-image inference in the app, so a
    picture uploaded by a reviewer goes through exactly the same steps the
    model saw at evaluation time.
    """
    return transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(cfg: TrainConfig) -> transforms.Compose:
    """Same standardization, plus mild augmentation.

    The augmentations are deliberately conservative: flips and small
    rotations are plausible variations of how a scan can be captured,
    whereas aggressive crops or distortions could remove or invent
    structure that matters.
    """
    aug = cfg.augmentation
    jitter = aug["color_jitter"]
    return transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]),
            transforms.RandomVerticalFlip(p=aug["vertical_flip"]),
            transforms.RandomRotation(degrees=aug["rotation_degrees"]),
            transforms.ColorJitter(brightness=jitter, contrast=jitter),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Turn one PIL image into a 1 x 3 x 224 x 224 batch ready for the model."""
    return build_eval_transform()(image).unsqueeze(0)


def _balanced_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """Oversample the minority class.

    Medical datasets are usually skewed toward benign; without this the
    model can reach high accuracy by never predicting malignant.
    """
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    class_weights = 1.0 / counts.clamp(min=1)
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Create the training and validation loaders described by ``cfg``."""
    train_dir = Path(cfg.data_dir) / "train"
    val_dir = Path(cfg.data_dir) / "val"

    for split_dir in (train_dir, val_dir):
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected an image folder at {split_dir}. See README.md for the "
                "directory layout, or run scripts/make_sample_data.py to generate "
                "a small synthetic dataset."
            )

    train_ds = ImageFolder(train_dir, transform=build_train_transform(cfg))
    val_ds = ImageFolder(val_dir, transform=build_eval_transform())

    if tuple(train_ds.classes) != CLASS_NAMES:
        raise ValueError(
            f"Expected class directories {CLASS_NAMES}, found {tuple(train_ds.classes)}."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=_balanced_sampler(train_ds),
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
