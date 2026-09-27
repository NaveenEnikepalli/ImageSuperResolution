"""Streamlit Frontend Application for Image Super-Resolution.

Author: Antigravity
Purpose: Interactive user interface supporting image uploading, resolution scale selection (2x, 4x, 8x),
before/after visual comparison, automatic model routing, and image download.
"""

import sys
import io
import os
import time
import logging
from pathlib import Path
from PIL import Image
import requests
import streamlit as st

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Backend service configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Image Super-Resolution",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_allowed_html=True,
)


def process_via_backend(image_bytes: bytes, scale: int) -> bytes:
    """Send image payload to FastAPI backend service for processing.

    Args:
        image_bytes (bytes): Input image payload.
        scale (int): Output scale factor (2, 4, or 8).

    Returns:
        bytes: Processed PNG image bytes.
    """
    url = f"{BACKEND_URL}/enhance"
    files = {"file": ("input_image.png", image_bytes, "image/png")}
    data = {"scale": scale}

    response = requests.post(url, files=files, data=data, timeout=120)
    if response.status_code == 200:
        return response.content
    else:
        raise RuntimeError(
            f"Backend request failed with status {response.status_code}: {response.text}"
        )


def process_via_local_inference(image_bytes: bytes, scale: int) -> bytes:
    """Fallback execution using direct local inference package.

    Args:
        image_bytes (bytes): Input image payload.
        scale (int): Output scale factor (2, 4, or 8).

    Returns:
        bytes: Processed PNG image bytes.
    """
    from inference.model_selector import select_model

    engine = select_model(scale=scale)
    input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output_pil = engine.enhance(input_pil)

    buf = io.BytesIO()
    output_pil.save(buf, format="PNG")
    return buf.getvalue()


def main():
    """Render Streamlit application UI."""
    st.markdown('<div class="main-title">✨ Image Super-Resolution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enhance your low-resolution images with AI-powered multi-scale super resolution (2x, 4x, 8x).</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Options
    with st.sidebar:
        st.header("⚙️ Settings")

        # Resolution Selection: User selects ONLY resolution scale
        scale_option = st.radio(
            "Select Output Scale Factor",
            options=["2x", "4x", "8x"],
            index=1,
            help="Choose desired resolution scale: 2x, 4x, or 8x image expansion.",
        )
        scale = int(scale_option.replace("x", ""))

        st.markdown("---")
        st.markdown("### ℹ️ Scale Details")
        if scale == 2:
            st.info("⚡ **2x Scaling**: Fast, high-fidelity restoration suitable for mild low-res images.")
        elif scale == 4:
            st.info("🔍 **4x Scaling**: Balanced deep restoration for standard images.")
        elif scale == 8:
            st.info("🚀 **8x Scaling**: Maximum resolution synthesis for ultra-high-definition output.")

    # Main Workspace layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a low-resolution image file...",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WEBP",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            input_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = input_img.size

            st.image(input_img, caption=f"Original Input Image ({w} × {h})", use_container_width=True)

            enhance_button = st.button("🚀 Enhance Image", type="primary", use_container_width=True)
        else:
            enhance_button = False

    with col2:
        st.subheader("✨ Enhanced Output")

        if uploaded_file is not None and enhance_button:
            with st.spinner(f"Enhancing image at {scale}x resolution... Please wait."):
                start_time = time.time()
                try:
                    # Attempt backend processing first
                    try:
                        output_bytes = process_via_backend(image_bytes, scale)
                        mode_used = "FastAPI Backend"
                    except Exception as backend_err:
                        st.info(f"Backend API unavailable ({backend_err}). Falling back to local inference...")
                        output_bytes = process_via_local_inference(image_bytes, scale)
                        mode_used = "Local Inference Engine"

                    elapsed_time = time.time() - start_time
                    output_img = Image.open(io.BytesIO(output_bytes))
                    out_w, out_h = output_img.size

                    st.image(
                        output_img,
                        caption=f"Enhanced Output ({out_w} × {out_h})",
                        use_container_width=True,
                    )

                    st.success(
                        f"Done in {elapsed_time:.2f}s using {mode_used}! "
                        f"Scaled from {w}×{h} to {out_w}×{out_h} ({scale}x)."
                    )

                    # Download button
                    st.download_button(
                        label=f"💾 Download Enhanced Image ({out_w}x{out_h})",
                        data=output_bytes,
                        file_name=f"enhanced_x{scale}_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to enhance image: {e}")
        else:
            st.info("Upload an image on the left panel and click 'Enhance Image' to view the output.")


if __name__ == "__main__":
    main()
