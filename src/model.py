"""EfficientNet-B0 classifier with a single-logit head.

The network outputs one raw logit per image. A sigmoid turns that logit
into P(malignant), and the caller compares it against an adjustable
threshold. Keeping the sigmoid out of the module means training can use
``BCEWithLogitsLoss``, which is numerically stabler than sigmoid + BCE.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import CLASS_NAMES


class CancerClassifier(nn.Module):
    """EfficientNet-B0 backbone, binary head, sigmoid probability."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        return self.backbone(x).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(malignant) in ``[0, 1]``, shape ``(batch,)``."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the pretrained feature extractor.

        The classifier head always stays trainable.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = trainable

    @property
    def target_layer(self) -> nn.Module:
        """Last convolutional block, the layer Grad-CAM hooks into."""
        return self.backbone.features[-1]


def label_from_probability(probability: float, threshold: float) -> str:
    """Map a probability and a threshold onto a class name."""
    return CLASS_NAMES[1] if probability >= threshold else CLASS_NAMES[0]


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[CancerClassifier, dict]:
    """Rebuild a trained model from disk.

    Returns the model in eval mode plus the metadata stored alongside the
    weights (threshold chosen at training time, validation metrics, and the
    image size the run was trained at).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CancerClassifier(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("metadata", {})
