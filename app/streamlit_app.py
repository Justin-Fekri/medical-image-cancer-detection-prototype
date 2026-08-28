"""Streamlit reviewer interface for the cancer detection prototype.

Design intent: a human stays in the loop. The app never hides the number
behind a verdict -- it shows the probability, the threshold that produced
the label, how close the two are, and a Grad-CAM overlay showing which
region drove the score. The educational-only framing is on screen at all
times, not buried in a README.
"""

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Allow `streamlit run app/streamlit_app.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig  # noqa: E402
from src.data import preprocess_pil  # noqa: E402
from src.interpretability import GradCAM, overlay_heatmap  # noqa: E402
from src.model import label_from_probability, load_checkpoint  # noqa: E402

CFG = TrainConfig()

st.set_page_config(
    page_title="Medical Image Cancer Detection Prototype",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def get_model(checkpoint_path: str):
    """Load the checkpoint once per session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    return model, metadata, device


@st.cache_data(show_spinner="Running model...")
def run_inference(image_bytes: bytes, checkpoint_path: str) -> tuple[float, Image.Image]:
    """Compute P(malignant) and the Grad-CAM overlay for one uploaded image.

    Cached on the raw bytes so that moving the threshold slider re-labels
    instantly instead of re-running the network -- the threshold is applied
    after inference, never inside it.
    """
    import io

    model, _, device = get_model(checkpoint_path)
    image = Image.open(io.BytesIO(image_bytes))

    tensor = preprocess_pil(image).to(device)
    probability = float(model.predict_proba(tensor)[0])

    with GradCAM(model) as cam:
        heatmap = cam(tensor.clone())
    return probability, overlay_heatmap(image, heatmap)


def render_disclaimer() -> None:
    st.warning(
        "**Educational and research use only — not a diagnostic device.** "
        "This prototype was built to demonstrate a deep learning pipeline. It has "
        "not been clinically validated, cleared by any regulator, or tested on a "
        "representative patient population. Nothing it outputs is a medical opinion. "
        "Do not upload identifiable patient data.",
        icon="⚠️",
    )


def render_sidebar(metadata: dict) -> float:
    st.sidebar.header("Reviewer controls")
    threshold = st.sidebar.slider(
        "Classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(metadata.get("suggested_threshold", CFG.decision_threshold)),
        step=0.01,
        help=(
            "An image is labelled malignant when P(malignant) is at or above this "
            "value. Lower it to catch more positives at the cost of more false "
            "alarms; raise it for the opposite trade-off."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Checkpoint")
    st.sidebar.write(f"**Architecture:** {metadata.get('architecture', 'efficientnet_b0')}")
    st.sidebar.write(f"**Input:** {metadata.get('image_size', 224)} x {metadata.get('image_size', 224)} RGB")
    if metadata.get("val_auroc") is not None:
        st.sidebar.write(f"**Validation AUROC:** {metadata['val_auroc']:.4f}")
    if metadata.get("suggested_threshold") is not None:
        st.sidebar.write(f"**Suggested threshold:** {metadata['suggested_threshold']:.2f}")
    if metadata.get("data_dir"):
        st.sidebar.caption(f"Trained on: `{metadata['data_dir']}`")
        st.sidebar.caption(
            "If this points at the bundled synthetic sample data, the metrics above "
            "describe geometric shapes, not medical images."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Grad-CAM highlights the region that most increased the malignant score. "
        "It explains where the network looked, which is not the same as explaining "
        "whether it was right."
    )
    return threshold


def render_result(probability: float, threshold: float) -> None:
    label = label_from_probability(probability, threshold)
    margin = abs(probability - threshold)

    left, right = st.columns(2)
    with left:
        st.metric("P(malignant)", f"{probability:.4f}")
        st.progress(probability)
    with right:
        st.metric("Predicted label", label.upper())
        st.metric("Distance from threshold", f"{margin:.4f}")

    if margin < 0.10:
        st.info(
            f"This score sits {margin:.3f} from the {threshold:.2f} threshold. "
            "Borderline cases like this flip label on small threshold changes and "
            "deserve a closer look rather than an automatic reading.",
            icon="ℹ️",
        )


def main() -> None:
    st.title("🔬 Medical Image Cancer Detection Prototype")
    st.caption(
        "EfficientNet-B0 · 224 x 224 RGB standardization · sigmoid probability · "
        "adjustable threshold · Grad-CAM interpretability"
    )
    render_disclaimer()

    checkpoint_path = CFG.checkpoint_path
    if not checkpoint_path.exists():
        st.error(f"No trained checkpoint found at `{checkpoint_path}`.")
        st.markdown("Generate the sample dataset and train a model first:")
        st.code(
            "python scripts/make_sample_data.py --n 400\n"
            "python -m src.train --epochs 5 --batch-size 16",
            language="bash",
        )
        st.stop()

    _, metadata, _ = get_model(str(checkpoint_path))
    threshold = render_sidebar(metadata)

    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    )
    if uploaded is None:
        st.info(
            "Upload an image to begin. Any image works — the model resizes it to "
            "224 x 224 RGB first. Try one from `data/test/` after generating the "
            "sample dataset.",
            icon="⬆️",
        )
        return

    image_bytes = uploaded.getvalue()
    probability, overlay = run_inference(image_bytes, str(checkpoint_path))

    st.subheader("Result")
    render_result(probability, threshold)

    st.subheader("Image and saliency")
    original_col, overlay_col = st.columns(2)
    with original_col:
        st.image(
            Image.open(uploaded).convert("RGB"),
            caption="Uploaded image (as provided)",
            use_container_width=True,
        )
    with overlay_col:
        st.image(
            overlay,
            caption="Grad-CAM overlay on the standardized 224 x 224 input",
            use_container_width=True,
        )

    with st.expander("How this number was produced"):
        st.markdown(
            f"""
1. The image was converted to RGB and resized to **224 x 224**, then normalized
   with ImageNet channel statistics.
2. EfficientNet-B0 produced a single raw logit, and a **sigmoid** mapped it to
   **P(malignant) = {probability:.4f}**.
3. That probability was compared against your threshold of **{threshold:.2f}**,
   giving **{label_from_probability(probability, threshold)}**.
4. Grad-CAM weighted the last convolutional feature map by the gradient of the
   logit to produce the overlay.

Changing the slider re-labels step 3 only. The probability itself does not
change, because the model already ran.
"""
        )


if __name__ == "__main__":
    main()
