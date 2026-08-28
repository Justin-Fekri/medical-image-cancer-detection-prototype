"""Single-image inference: probability, label, and an optional saliency map."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .config import TrainConfig
from .data import preprocess_pil
from .interpretability import GradCAM, overlay_heatmap
from .model import CancerClassifier, label_from_probability, load_checkpoint


@dataclass
class Prediction:
    """One model output, kept together with the threshold that produced it."""

    probability: float  # P(malignant)
    threshold: float
    label: str
    overlay: Image.Image | None = None

    @property
    def confidence(self) -> float:
        """Probability of whichever class was reported."""
        return (
            self.probability
            if self.label == "malignant"
            else 1.0 - self.probability
        )


def predict_image(
    model: CancerClassifier,
    image: Image.Image,
    threshold: float = 0.5,
    device: torch.device | str = "cpu",
    explain: bool = True,
) -> Prediction:
    """Run one image through the standardization pipeline and the model."""
    tensor = preprocess_pil(image).to(device)

    probability = float(model.predict_proba(tensor)[0])
    label = label_from_probability(probability, threshold)

    overlay = None
    if explain:
        with GradCAM(model) as cam:
            heatmap = cam(tensor.clone())
        overlay = overlay_heatmap(image, heatmap)

    return Prediction(
        probability=probability,
        threshold=threshold,
        label=label,
        overlay=overlay,
    )


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Classify a single image (educational / research use only)."
    )
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--threshold", type=float, default=cfg.decision_threshold)
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=None,
        help="Write the Grad-CAM overlay to this path.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(args.checkpoint, device=device)

    image = Image.open(args.image)
    result = predict_image(
        model,
        image,
        threshold=args.threshold,
        device=device,
        explain=args.save_overlay is not None,
    )

    print(f"image:            {args.image}")
    print(f"P(malignant):     {result.probability:.4f}")
    print(f"threshold:        {result.threshold:.2f}")
    print(f"predicted label:  {result.label}")
    print(f"confidence:       {result.confidence:.4f}")
    if metadata.get("val_auroc") is not None:
        print(f"checkpoint AUROC: {metadata['val_auroc']:.4f} (validation)")
    print("\nEducational and research use only. Not a diagnostic device.")

    if args.save_overlay and result.overlay is not None:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        result.overlay.save(args.save_overlay)
        print(f"overlay written to {args.save_overlay}")


if __name__ == "__main__":
    main()
