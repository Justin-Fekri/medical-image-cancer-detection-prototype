"""Train the EfficientNet-B0 binary head on a folder of images or synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from meddetect.config import DEFAULT_CHECKPOINT, INPUT_SIZE
from meddetect.model import build_model, save_checkpoint
from meddetect.preprocess import image_to_tensor
from meddetect.synthetic import generate_labeled_images


class FolderBinaryDataset(Dataset):
    """Expect ``root/negative/*.png`` and ``root/positive/*.png`` (or jpg)."""

    def __init__(self, root: str | Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        root_path = Path(root)
        mapping = {"negative": 0, "positive": 1, "benign": 0, "malignant": 1}
        for name, label in mapping.items():
            folder = root_path / name
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {root_path}. "
                "Create negative/ and positive/ subfolders."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor, _ = image_to_tensor(image)
        return tensor, torch.tensor([label], dtype=torch.float32)


class SyntheticDataset(Dataset):
    def __init__(self, n_per_class: int, seed: int) -> None:
        self.pairs = generate_labeled_images(n_per_class=n_per_class, size=256, seed=seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.pairs[index]
        tensor, _ = image_to_tensor(image)
        return tensor, torch.tensor([label], dtype=torch.float32)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return (preds == targets).float().mean().item()


def train(
    *,
    data_dir: str | Path | None,
    synthetic: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    freeze_backbone: bool,
    output: Path,
    pretrained: bool,
    n_synthetic: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    torch.manual_seed(seed)
    resolved = torch.device(device)
    if synthetic:
        dataset = SyntheticDataset(n_per_class=n_synthetic, seed=seed)
    else:
        if data_dir is None:
            raise ValueError("Provide --data-dir or pass --synthetic.")
        dataset = FolderBinaryDataset(data_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = build_model(pretrained=pretrained)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.to(resolved)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    criterion = nn.BCEWithLogitsLoss()
    history: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_acc = 0.0
        steps = 0
        for images, targets in loader:
            images = images.to(resolved)
            targets = targets.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy_from_logits(logits.detach(), targets)
            steps += 1
        history = {
            "epoch": float(epoch),
            "loss": running_loss / max(steps, 1),
            "accuracy": running_acc / max(steps, 1),
        }
        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"loss={history['loss']:.4f}  acc={history['accuracy']:.3f}"
        )

    model.eval()
    save_checkpoint(
        model,
        output,
        trained_on="synthetic" if synthetic else str(data_dir),
        freeze_backbone=freeze_backbone,
        epochs=epochs,
        input_size=INPUT_SIZE,
        metrics=history,
        notes=(
            "Educational and research-only prototype. "
            "This checkpoint is not validated for clinical use."
        ),
    )
    print(f"saved checkpoint -> {output}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the EfficientNet-B0 prototype (educational / research only)."
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder with negative/ and positive/ images.")
    parser.add_argument("--synthetic", action="store_true", help="Train on generated images (no patient data).")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-synthetic", type=int, default=24, help="Images per class when using --synthetic.")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--from-scratch", action="store_true", help="Do not load ImageNet backbone weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_dir=args.data_dir,
        synthetic=args.synthetic or args.data_dir is None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=not args.unfreeze_backbone,
        output=args.output,
        pretrained=not args.from_scratch,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
