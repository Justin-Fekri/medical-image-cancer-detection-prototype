import torch

from meddetect.inference import apply_threshold
from meddetect.model import build_model, last_feature_layer


def test_efficientnet_b0_emits_one_logit() -> None:
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 1)
    probs = torch.sigmoid(logits)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)


def test_last_feature_layer_is_present() -> None:
    model = build_model(pretrained=False)
    layer = last_feature_layer(model)
    assert layer is model.features[-1]


def test_threshold_flags_at_boundary() -> None:
    on_cutoff = apply_threshold(0.50, 0.50)
    assert on_cutoff.predicted_positive is True
    below = apply_threshold(0.499, 0.50)
    assert below.predicted_positive is False
    above = apply_threshold(0.91, 0.80)
    assert above.predicted_positive is True


def test_threshold_rejects_invalid_probability() -> None:
    import pytest

    with pytest.raises(ValueError):
        apply_threshold(1.2, 0.5)
    with pytest.raises(ValueError):
        apply_threshold(0.2, -0.1)
