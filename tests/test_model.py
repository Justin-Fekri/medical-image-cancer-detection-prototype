"""Shape and probability guarantees for the classifier head."""

import torch

from src.config import IMAGE_SIZE
from src.model import CancerClassifier, label_from_probability


def _model() -> CancerClassifier:
    # pretrained=False keeps the test offline and fast.
    return CancerClassifier(pretrained=False)


def test_forward_returns_one_logit_per_image():
    logits = _model()(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (4,)


def test_predict_proba_is_bounded():
    # Deliberately extreme inputs, to check the sigmoid actually bounds output.
    probabilities = _model().predict_proba(
        torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE) * 50
    )
    assert probabilities.shape == (8,)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)


def test_freezing_backbone_leaves_head_trainable():
    model = _model()
    model.set_backbone_trainable(False)

    assert not any(p.requires_grad for p in model.backbone.features.parameters())
    assert all(p.requires_grad for p in model.backbone.classifier.parameters())

    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone.features.parameters())


def test_threshold_decides_the_label():
    assert label_from_probability(0.80, threshold=0.5) == "malignant"
    assert label_from_probability(0.80, threshold=0.9) == "benign"
    # A probability exactly at the threshold counts as positive.
    assert label_from_probability(0.30, threshold=0.30) == "malignant"
