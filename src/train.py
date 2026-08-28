"""Fine-tune EfficientNet-B0 on the benign / malignant split.

Two-phase schedule: the pretrained feature extractor stays frozen while the
fresh head learns, then everything unfreezes for the remaining epochs. This
keeps the randomly initialized head from washing out the pretrained filters
with large early gradients.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import build_dataloaders
from .metrics import auroc, best_threshold_for_sensitivity, classification_report
from .model import CancerClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple silicon; falls back to CPU when the build lacks MPS.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: CancerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over the training set, returning mean loss."""
    model.train()
    running_loss = 0.0
    seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        # BCEWithLogitsLoss expects float targets shaped like the logits.
        targets = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        seen += images.size(0)

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate_split(
    model: CancerClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mean loss plus the raw labels and probabilities for the split."""
    model.eval()
    running_loss = 0.0
    seen = 0
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        targets = labels.float().to(device, non_blocking=True)

        logits = model(images)
        running_loss += criterion(logits, targets).item() * images.size(0)
        seen += images.size(0)

        all_labels.append(labels.numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return (
        running_loss / max(seen, 1),
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def save_checkpoint(
    cfg: TrainConfig,
    state: dict,
    labels: np.ndarray,
    probabilities: np.ndarray,
    epoch: int,
    val_auroc: float,
    epochs_run: int,
    frozen_backbone: bool,
) -> dict:
    """Write weights plus the metadata the app and CLI read back.

    Called every time validation AUROC improves, not only at the end, so a
    run that is interrupted still leaves a usable checkpoint on disk.
    """
    suggested = best_threshold_for_sensitivity(
        labels, probabilities, min_sensitivity=0.90
    )
    metadata = {
        "image_size": 224,
        "class_names": ["benign", "malignant"],
        "architecture": "efficientnet_b0",
        "best_epoch": epoch,
        "val_auroc": val_auroc,
        "suggested_threshold": suggested,
        "val_report_at_suggested_threshold": classification_report(
            labels, probabilities, suggested
        ).as_dict(),
        "epochs_run": epochs_run,
        "frozen_backbone": frozen_backbone,
        "data_dir": str(cfg.data_dir),
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "metadata": metadata}, cfg.checkpoint_path)
    return metadata


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = resolve_device()
    print(f"device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"train images: {len(train_loader.dataset)}  "
        f"val images: {len(val_loader.dataset)}"
    )

    model = CancerClassifier(pretrained=True).to(device)
    # In linear-probe mode the backbone never unfreezes: only the head trains.
    # Backpropagating through EfficientNet-B0 costs roughly 4x a head-only step
    # on CPU, so this is the difference between minutes and hours on a laptop.
    model.set_backbone_trainable(False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: list[dict] = []
    best_auroc = -1.0
    best_state: dict | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        # Unfreeze the backbone once the head has settled, and rebuild the
        # optimizer so the newly trainable parameters are actually stepped.
        if not cfg.freeze_backbone and epoch == cfg.warmup_frozen_epochs + 1:
            print("unfreezing backbone")
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate / 10,
                weight_decay=cfg.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(cfg.epochs - epoch + 1, 1)
            )

        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_labels, val_probabilities = evaluate_split(
            model, val_loader, criterion, device
        )
        scheduler.step()

        val_auroc = auroc(val_labels, val_probabilities)
        report = classification_report(
            val_labels, val_probabilities, cfg.decision_threshold
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_accuracy": report.accuracy,
                "val_sensitivity": report.sensitivity,
                "val_specificity": report.specificity,
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"epoch {epoch:>3}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_auroc={val_auroc:.4f}  val_acc={report.accuracy:.4f}  "
            f"({history[-1]['seconds']}s)"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            save_checkpoint(
                cfg,
                best_state,
                val_labels,
                val_probabilities,
                epoch=epoch,
                val_auroc=val_auroc,
                epochs_run=len(history),
                frozen_backbone=cfg.freeze_backbone,
            )
            print(f"  saved checkpoint (val_auroc improved to {val_auroc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"early stopping: no AUROC gain in {cfg.patience} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; did the loaders yield data?")

    # Score the best weights once more so the suggested threshold matches the
    # weights actually being saved, not whatever the last epoch produced.
    model.load_state_dict(best_state)
    _, val_labels, val_probabilities = evaluate_split(
        model, val_loader, criterion, device
    )
    metadata = save_checkpoint(
        cfg,
        best_state,
        val_labels,
        val_probabilities,
        epoch=best_epoch,
        val_auroc=best_auroc,
        epochs_run=len(history),
        frozen_backbone=cfg.freeze_backbone,
    )
    suggested = metadata["suggested_threshold"]
    final_report = classification_report(val_labels, val_probabilities, suggested)

    history_path = cfg.checkpoint_path.parent / "training_history.json"
    history_path.write_text(
        json.dumps({"history": history, "metadata": metadata}, indent=2)
    )

    print(f"\nbest epoch {best_epoch}  val AUROC {best_auroc:.4f}")
    print(f"suggested threshold (sensitivity >= 0.90): {suggested:.2f}")
    print(f"  {final_report}")
    print(f"checkpoint: {cfg.checkpoint_path}")
    print(f"history:    {history_path}")
    return metadata


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the cancer detection prototype.")
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--warmup-frozen-epochs", type=int, default=cfg.warmup_frozen_epochs)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Linear-probe mode: train only the head. Much faster on CPU.",
    )
    args = parser.parse_args()

    cfg.data_dir = args.data_dir
    cfg.checkpoint_path = args.checkpoint
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.warmup_frozen_epochs = args.warmup_frozen_epochs
    cfg.num_workers = args.num_workers
    cfg.patience = args.patience
    cfg.seed = args.seed
    cfg.freeze_backbone = args.freeze_backbone
    return cfg


if __name__ == "__main__":
    train(parse_args())
