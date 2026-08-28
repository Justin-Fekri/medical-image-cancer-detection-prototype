"""Streamlit workbench for the EfficientNet-B0 classification prototype."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from meddetect.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_THRESHOLD,
    DISCLAIMER,
    INPUT_SIZE,
    SAMPLES_DIR,
)
from meddetect.inference import apply_threshold, run_inference
from meddetect.model import load_checkpoint
from meddetect.preprocess import ImageLoadError, load_rgb_image

PAGE_TITLE = "Medical Image Classification Prototype"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1200px; }
        .disclaimer-banner {
            background: #FEF3C7;
            border: 1px solid #F59E0B;
            color: #78350F;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .result-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        .flag-on {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            color: #991B1B;
            padding: 0.7rem 0.9rem;
            border-radius: 10px;
            font-weight: 650;
        }
        .flag-off {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            color: #065F46;
            padding: 0.7rem 0.9rem;
            border-radius: 10px;
            font-weight: 650;
        }
        .muted { color: #475569; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _list_samples() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


@st.cache_resource(show_spinner="Loading EfficientNet-B0…")
def _cached_model() -> tuple[object, dict]:
    model, metadata = load_checkpoint(DEFAULT_CHECKPOINT, pretrained_fallback=True)
    model.eval()
    return model, metadata


def _source_from_widgets(samples: list[Path]) -> tuple[Image.Image | None, str | None]:
    uploaded = st.file_uploader(
        "Upload a medical image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        help="The image is converted to 224×224 RGB before scoring. Do not upload identifiable patient photos to public demos.",
    )
    sample_names = ["— none —"] + [path.name for path in samples]
    choice = st.selectbox(
        "Or open a synthetic sample (no patient data)",
        sample_names,
        index=0,
    )

    if uploaded is not None:
        try:
            return load_rgb_image(uploaded.getvalue()), uploaded.name
        except ImageLoadError as exc:
            st.error(str(exc))
            return None, None

    if choice != "— none —":
        path = SAMPLES_DIR / choice
        return load_rgb_image(path), path.name

    return None, None


def _render_empty_state() -> None:
    st.info("Upload an image or choose a synthetic sample to run the prototype.")
    st.markdown(
        """
        **What this screen will show**
        - Standardized **224×224 RGB** preview
        - EfficientNet-B0 **sigmoid probability**
        - Your **classification threshold** applied as a review flag
        - A **Grad-CAM** overlay so a reviewer can see which regions influenced the score
        """
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧬", layout="wide")
    _inject_styles()

    st.title("Medical Image Classification and Grad-CAM Prototype")
    st.caption("Python · PyTorch · EfficientNet-B0 · Streamlit · 2025")
    st.markdown(f'<div class="disclaimer-banner">{DISCLAIMER}</div>', unsafe_allow_html=True)

    samples = _list_samples()
    model, metadata = _cached_model()
    source_tag = str(metadata.get("source", "unknown"))
    trained_on = str(metadata.get("trained_on", metadata.get("notes", "see checkpoint metadata")))

    with st.sidebar:
        st.header("Reviewer controls")
        threshold = st.slider(
            "Classification threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_THRESHOLD),
            step=0.01,
            help="Flag the case when the sigmoid probability is at or above this cutoff.",
        )
        st.caption(
            "Lower thresholds flag more images (higher sensitivity). "
            "Higher thresholds flag fewer images (higher specificity)."
        )
        st.divider()
        st.subheader("Model")
        st.write("Architecture: **EfficientNet-B0**")
        st.write("Input: **224×224 RGB**")
        st.write("Head: **1 logit → sigmoid**")
        st.write(f"Weights: `{Path(source_tag).name}`")
        st.caption(f"Training source: {trained_on}")
        st.divider()
        st.warning(
            "Not a medical device. Outputs are for education and research only."
        )

    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.subheader("Input")
        image, filename = _source_from_widgets(samples)

    if image is None:
        with right:
            st.subheader("Result")
            _render_empty_state()
        st.stop()

    image_key = f"{filename}-{image.size}-{image.mode}"
    if st.session_state.get("image_key") != image_key:
        try:
            with st.spinner("Running EfficientNet-B0 and Grad-CAM…"):
                result = run_inference(model, image)
        except Exception as exc:  # noqa: BLE001 — show a reviewer-facing error
            st.error(f"Inference failed: {exc}")
            st.stop()
        st.session_state["image_key"] = image_key
        st.session_state["result"] = result
        st.session_state["filename"] = filename

    result = st.session_state["result"]
    decision = apply_threshold(result.probability, threshold)

    with left:
        st.image(image, caption=f"Original · {filename} · {result.original_size[0]}×{result.original_size[1]}", use_container_width=True)
        st.image(
            result.preview_rgb,
            caption=f"Preprocessed · {INPUT_SIZE}×{INPUT_SIZE} RGB (model input)",
            use_container_width=True,
        )

    with right:
        st.subheader("Result")
        flag_class = "flag-on" if decision.predicted_positive else "flag-off"
        st.markdown(
            f'<div class="{flag_class}">{decision.label}</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(3)
        metric_cols[0].metric("Sigmoid probability", f"{result.probability:.3f}")
        metric_cols[1].metric("Threshold", f"{decision.threshold:.2f}")
        metric_cols[2].metric("Logit", f"{result.logit:.3f}")
        st.progress(min(max(result.probability, 0.0), 1.0))
        st.caption("Probability of the positive class. This is a model score, not a diagnosis.")

        st.markdown("##### Visual interpretability (Grad-CAM)")
        cam_cols = st.columns(2)
        cam_cols[0].image(result.overlay, caption="Overlay on 224×224 input", use_container_width=True)
        cam_cols[1].image(result.heatmap_color, caption="Class activation heatmap", use_container_width=True)
        st.markdown(
            '<p class="muted">Warmer regions contributed more to the positive-class logit. '
            "Grad-CAM is an attribution aid for reviewers, not proof of a lesion or a diagnosis.</p>",
            unsafe_allow_html=True,
        )

    with st.expander("Technical details for reviewers"):
        st.write(
            {
                "architecture": "EfficientNet-B0",
                "framework": "PyTorch",
                "input": f"{INPUT_SIZE}x{INPUT_SIZE} RGB, ImageNet mean/std",
                "output": "binary logit with sigmoid probability",
                "threshold": decision.threshold,
                "probability": round(result.probability, 6),
                "predicted_positive": decision.predicted_positive,
                "checkpoint": source_tag,
                "disclaimer": DISCLAIMER,
            }
        )
        st.markdown(
            "Retrain on your own labeled folders with `python train.py --data-dir path/to/data`. "
            "The bundled checkpoint is **synthetic-only** unless you replace it."
        )


if __name__ == "__main__":
    main()
