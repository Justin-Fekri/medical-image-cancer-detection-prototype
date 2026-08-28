"""EfficientNet-B0 binary classifier with a sigmoid probability head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import ARCH_NAME, DEFAULT_CHECKPOINT, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def build_model(pretrained: bool = False, num_outputs: int = 1) -> nn.Module:
    """Build EfficientNet-B0 with a single logit for binary classification.

    The ImageNet classifier is replaced by a linear layer that emits one logit.
    Callers apply ``torch.sigmoid`` to obtain a probability in [0, 1].
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


def last_feature_layer(model: nn.Module) -> nn.Module:
    """Return the final convolutional block used for Grad-CAM."""
    return model.features[-1]


def load_checkpoint(
    path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    pretrained_fallback: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved prototype checkpoint, or ImageNet-initialized weights.

    When ``path`` is missing, the default checkpoint is used if present.
    Otherwise the backbone is initialized from ImageNet (if requested) and the
    classification head is random — still not a clinical model.
    """
    checkpoint_path = Path(path) if path else DEFAULT_CHECKPOINT
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model = build_model(pretrained=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state)
        metadata = payload if isinstance(payload, dict) else {"state_dict": payload}
        metadata.setdefault("source", str(checkpoint_path))
        metadata.setdefault("arch", ARCH_NAME)
        return model, metadata

    model = build_model(pretrained=pretrained_fallback)
    metadata = {
        "arch": ARCH_NAME,
        "source": "imagenet_backbone_random_head" if pretrained_fallback else "random",
        "notes": "No prototype checkpoint found. Weights are not medically trained.",
        "input_size": INPUT_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    return model, metadata


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    **metadata: Any,
) -> Path:
    """Persist model weights plus reviewer-facing metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": ARCH_NAME,
        "input_size": INPUT_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "disclaimer": (
            "Educational and research-only prototype. Not a diagnostic device."
        ),
        **metadata,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, destination)
    return destination
