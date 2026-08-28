# Medical Image Cancer Detection Prototype

Educational and research-only CNN prototype (2025). **This is not a diagnostic device** and must not be used to screen, diagnose, treat, or rule out cancer.

The app takes a medical image, standardizes it to **224×224 RGB**, scores it with **EfficientNet-B0** (PyTorch), and shows a **sigmoid probability** against an **adjustable classification threshold**. **Grad-CAM** overlays highlight regions that influenced the score so a reviewer can inspect the decision.

Read [DISCLAIMER.md](DISCLAIMER.md) before running anything.

## What you get

- Streamlit workbench for upload + synthetic samples
- EfficientNet-B0 binary head (`1` logit → sigmoid)
- Adjustable review threshold (default `0.50`)
- Grad-CAM heatmap and overlay
- Training script for your own `negative/` / `positive/` folders
- A bundled checkpoint trained **only on synthetic images** so the UI works out of the box

The bundled weights are **not** a clinical model. They exist so GitHub clones can run the interface immediately.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 43127 --server.address 0.0.0.0
```

Open `http://127.0.0.1:43127`. Upload a PNG/JPEG (or pick a file under `samples/`). Move the **Classification threshold** slider to see how the review flag changes without re-running the network.

CPU is enough. A GPU is optional (`python train.py --device cuda`).

## Train on your own images

```
data/
  negative/   # or benign/
  positive/   # or malignant/
```

```bash
python train.py --data-dir data --epochs 10 --output checkpoints/prototype_efficientnet_b0.pt
```

Regenerate the synthetic demo checkpoint:

```bash
python train.py --synthetic --epochs 8 --n-synthetic 24 --output checkpoints/prototype_efficientnet_b0.pt
```

By default the EfficientNet-B0 backbone is initialized from ImageNet and frozen; only the binary head is trained. Pass `--unfreeze-backbone` to fine-tune the convolutional features.

## Tests

```bash
python -m pytest -q
```

## Project layout

```
app.py                 Streamlit interface
train.py               Binary training entry point
meddetect/             Model, preprocess, Grad-CAM, inference
samples/               Synthetic PNGs (no patient data)
checkpoints/           Prototype EfficientNet-B0 weights
tests/                 Preprocess, threshold, Grad-CAM checks
```

## Design notes

| Piece | Choice |
| --- | --- |
| Architecture | EfficientNet-B0 (torchvision) |
| Input | 224×224 RGB, ImageNet mean/std |
| Output | Single logit + sigmoid probability |
| Decision | `probability >= threshold` |
| Interpretability | Grad-CAM on the last feature block |

## License

MIT. The license does not grant permission to use this software as a medical device.
