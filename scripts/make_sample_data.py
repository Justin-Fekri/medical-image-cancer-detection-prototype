#!/usr/bin/env python3
"""Generate a small synthetic dataset so the pipeline runs without a download.

These images are NOT medical images. They are procedurally drawn geometric
blobs: the "benign" class gets smooth, round, low-contrast shapes and the
"malignant" class gets irregular, spiculated, higher-contrast ones. The
distinction is deliberately learnable, which is the whole point -- it lets a
reviewer confirm that data loading, training, thresholding and Grad-CAM all
work end to end on a laptop CPU.

Any accuracy or AUROC measured on this data says something about the
plumbing and nothing whatsoever about medical performance. Replace this
directory with a real dataset before drawing any other conclusion.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANVAS = 256  # generated a bit larger than 224 so the resize has something to do
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _background(rng: np.random.Generator) -> Image.Image:
    """A soft grayscale gradient standing in for surrounding tissue."""
    base = rng.integers(60, 110)
    gradient = np.linspace(0, rng.integers(20, 50), CANVAS, dtype=np.float32)
    field = base + gradient[None, :] + gradient[:, None] * 0.5
    field += rng.normal(0, 6, size=(CANVAS, CANVAS))
    arr = np.clip(field, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _blob_points(
    rng: np.random.Generator, malignant: bool
) -> list[tuple[float, float]]:
    """Polygon outline of the lesion, in polar coordinates.

    Benign lesions get a nearly constant radius (smooth, well-circumscribed
    margin). Malignant ones get per-vertex radial noise plus occasional long
    spikes, imitating a spiculated margin.
    """
    cx, cy = rng.integers(90, 166, size=2)
    mean_radius = rng.integers(30, 55)
    n_vertices = 60

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    if malignant:
        wobble = rng.normal(0, mean_radius * 0.18, size=n_vertices)
        spikes = np.zeros(n_vertices)
        for _ in range(rng.integers(4, 9)):
            idx = rng.integers(0, n_vertices)
            spikes[idx] = rng.uniform(0.35, 0.75) * mean_radius
        radii = mean_radius + wobble + spikes
    else:
        radii = mean_radius + rng.normal(0, mean_radius * 0.03, size=n_vertices)

    radii = np.clip(radii, 8, 110)
    return [
        (float(cx + r * np.cos(a)), float(cy + r * np.sin(a)))
        for r, a in zip(radii, angles)
    ]


def make_image(rng: np.random.Generator, malignant: bool) -> Image.Image:
    """Draw one synthetic sample."""
    image = _background(rng)
    draw = ImageDraw.Draw(image)

    # Malignant blobs are brighter than their surroundings and more textured.
    fill = int(rng.integers(170, 225)) if malignant else int(rng.integers(120, 160))
    draw.polygon(_blob_points(rng, malignant), fill=(fill, fill, fill))

    if malignant:
        # Speckle inside the lesion: heterogeneous interior.
        arr = np.asarray(image).astype(np.float32)
        speckle = rng.normal(0, 18, size=arr.shape[:2])[..., None]
        image = Image.fromarray(np.clip(arr + speckle, 0, 255).astype(np.uint8))

    blur = rng.uniform(0.5, 1.2) if malignant else rng.uniform(1.5, 2.8)
    return image.filter(ImageFilter.GaussianBlur(radius=blur))


def generate(output_dir: Path, n_per_class: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}

    for label, malignant in (("benign", False), ("malignant", True)):
        # Assign each image to a split up front so the proportions are exact.
        assignments: list[str] = []
        for split, fraction in SPLITS.items():
            assignments += [split] * int(round(n_per_class * fraction))
        assignments += ["train"] * (n_per_class - len(assignments))
        rng.shuffle(assignments)

        for index, split in enumerate(assignments):
            split_dir = output_dir / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            make_image(rng, malignant).save(split_dir / f"{label}_{index:05d}.png")
            counts[f"{split}/{label}"] = counts.get(f"{split}/{label}", 0) + 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n", type=int, default=400, help="Images per class (default: 400)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    counts = generate(args.output_dir, args.n, args.seed)

    print(f"Wrote synthetic images to {args.output_dir}")
    for key in sorted(counts):
        print(f"  {key:<20} {counts[key]}")
    print(
        "\nReminder: synthetic geometric shapes, not medical images. "
        "Metrics measured here test the pipeline, not clinical accuracy."
    )


if __name__ == "__main__":
    main()
