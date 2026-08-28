"""Score a trained checkpoint on the held-out test split.

Prints a threshold sweep rather than a single accuracy number, because the
operating point is a choice a reviewer makes, not a property of the model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .config import TrainConfig
from .data import build_eval_transform
from .metrics import auroc, classification_report, sweep_thresholds
from .model import load_checkpoint
from .train import evaluate_split, resolve_device


def build_test_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"No test split at {test_dir}. Run scripts/make_sample_data.py first."
        )
    dataset = ImageFolder(test_dir, transform=build_eval_transform())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def print_sweep(labels: np.ndarray, probabilities: np.ndarray) -> None:
    header = (
        f"{'thresh':>7} {'acc':>7} {'sens':>7} {'spec':>7} "
        f"{'prec':>7} {'f1':>7} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    )
    print(header)
    print("-" * len(header))
    for report in sweep_thresholds(labels, probabilities):
        print(
            f"{report.threshold:>7.2f} {report.accuracy:>7.4f} "
            f"{report.sensitivity:>7.4f} {report.specificity:>7.4f} "
            f"{report.precision:>7.4f} {report.f1:>7.4f} "
            f"{report.true_positives:>5} {report.false_positives:>5} "
            f"{report.true_negatives:>5} {report.false_negatives:>5}"
        )


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Evaluate on the test split.")
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Operating point to report. Defaults to the checkpoint's suggestion.",
    )
    args = parser.parse_args()

    device = resolve_device()
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    threshold = args.threshold or metadata.get("suggested_threshold", 0.5)

    loader = build_test_loader(args.data_dir, args.batch_size, args.num_workers)
    _, labels, probabilities = evaluate_split(
        model, loader, nn.BCEWithLogitsLoss(), device
    )

    test_auroc = auroc(labels, probabilities)
    report = classification_report(labels, probabilities, threshold)

    print(f"checkpoint:   {args.checkpoint}")
    print(f"test images:  {len(labels)}")
    print(f"test AUROC:   {test_auroc:.4f}")
    print(f"\nAt the selected threshold {threshold:.2f}:")
    print(f"  {report}")
    print(
        f"  confusion: TP={report.true_positives} FP={report.false_positives} "
        f"TN={report.true_negatives} FN={report.false_negatives}"
    )
    print("\nThreshold sweep:")
    print_sweep(labels, probabilities)

    output = Path(args.checkpoint).parent / "test_metrics.json"
    output.write_text(
        json.dumps(
            {
                "test_auroc": test_auroc,
                "n_images": int(len(labels)),
                "selected_threshold": float(threshold),
                "report_at_selected_threshold": report.as_dict(),
                "sweep": [r.as_dict() for r in sweep_thresholds(labels, probabilities)],
                "checkpoint_metadata": metadata,
            },
            indent=2,
        )
    )
    print(f"\nwrote {output}")
    print(
        "Educational and research use only. These numbers describe this dataset "
        "and this pipeline, not clinical performance."
    )


if __name__ == "__main__":
    main()
