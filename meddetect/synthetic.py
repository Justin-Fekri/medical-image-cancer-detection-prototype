"""Synthetic dermoscopic-style images for demos and smoke-test training.

These images are not medical data. They exist so the prototype can be run
and trained without patient photographs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _skin_canvas(size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array(
        [
            rng.integers(175, 220),
            rng.integers(120, 175),
            rng.integers(95, 145),
        ],
        dtype=np.float32,
    )
    yy, xx = np.mgrid[0:size, 0:size]
    lighting = 1.0 - 0.18 * ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size**2)
    noise = rng.normal(0, 7.5, (size, size, 3))
    canvas = base * lighting[..., None] + noise
    # Faint scattered specks that resemble pores / small nevi.
    for _ in range(int(rng.integers(8, 18))):
        cy, cx = rng.integers(10, size - 10, size=2)
        radius = int(rng.integers(1, 3))
        shade = rng.normal(-18, 6, 3)
        canvas[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1] += shade
    return np.clip(canvas, 0, 255)


def _blob_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    cx, cy = rng.integers(size * 0.32, size * 0.68, size=2)
    for _ in range(int(rng.integers(4, 8))):
        dx = int(rng.integers(-28, 29))
        dy = int(rng.integers(-28, 29))
        rx = int(rng.integers(18, 52))
        ry = int(rng.integers(16, 48))
        bbox = [int(cx + dx - rx), int(cy + dy - ry), int(cx + dx + rx), int(cy + dy + ry)]
        draw.ellipse(bbox, fill=255)
    image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.5, 5.0))))
    return np.asarray(image, dtype=np.float32) / 255.0


def render_negative(size: int = 256, seed: int = 0) -> Image.Image:
    """Mostly uniform skin-like field without a dominant lesion."""
    canvas = _skin_canvas(size, _rng(seed))
    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def render_positive(size: int = 256, seed: int = 0) -> Image.Image:
    """Irregular darker blob with an uneven border (synthetic 'lesion')."""
    rng = _rng(seed + 10_000)
    canvas = _skin_canvas(size, rng)
    mask = _blob_mask(size, rng)[..., None]
    lesion_color = np.array(
        [
            rng.integers(40, 95),
            rng.integers(20, 70),
            rng.integers(15, 55),
        ],
        dtype=np.float32,
    )
    rim = np.array([rng.integers(110, 160), rng.integers(40, 80), rng.integers(35, 70)], dtype=np.float32)
    edge = np.clip(mask * 4.0, 0, 1) * (1.0 - np.clip(mask * 1.6, 0, 1))
    canvas = canvas * (1.0 - mask * 0.92) + lesion_color * mask * 0.92
    canvas = canvas * (1.0 - edge * 0.45) + rim * edge * 0.45
    speckle = rng.normal(0, 10, canvas.shape) * mask
    canvas = np.clip(canvas + speckle, 0, 255)
    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def generate_labeled_images(
    n_per_class: int,
    size: int = 256,
    seed: int = 0,
) -> list[tuple[Image.Image, int]]:
    """Return (image, label) pairs: 0 = negative, 1 = positive."""
    pairs: list[tuple[Image.Image, int]] = []
    for i in range(n_per_class):
        pairs.append((render_negative(size=size, seed=seed + i), 0))
        pairs.append((render_positive(size=size, seed=seed + i), 1))
    return pairs


def write_sample_gallery(directory: str | Path, size: int = 256) -> list[Path]:
    """Write a small set of clearly labeled synthetic demo images."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    specs = [
        ("synthetic_clear_01.png", lambda: render_negative(size=size, seed=0)),
        ("synthetic_clear_02.png", lambda: render_negative(size=size, seed=1)),
        ("synthetic_clear_03.png", lambda: render_negative(size=size, seed=5)),
        ("synthetic_lesion_01.png", lambda: render_positive(size=size, seed=0)),
        ("synthetic_lesion_02.png", lambda: render_positive(size=size, seed=1)),
        ("synthetic_lesion_03.png", lambda: render_positive(size=size, seed=5)),
    ]
    for name, factory in specs:
        path = out / name
        factory().save(path, format="PNG")
        written.append(path)
    readme = out / "README.md"
    readme.write_text(
        "# Synthetic sample images\n\n"
        "These PNG files are computer-generated and contain **no patient data**.\n"
        "Use them only to try the Streamlit prototype.\n",
        encoding="utf-8",
    )
    return written
