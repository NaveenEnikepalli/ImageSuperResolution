"""Inference Service Module for Backend FastAPI Application.

Author: Antigravity
Purpose: Bridge incoming HTTP image enhancement requests to the underlying
super-resolution inference engines (Student x2, Student x4, SwinIR x8).
"""

import io
import logging
from typing import Dict, Any, Union
from PIL import Image

from inference.model_selector import select_model

logger = logging.getLogger(__name__)

# Engine instance cache to avoid re-instantiating models per request
_ENGINE_CACHE: Dict[int, Any] = {}


def get_cached_engine(scale: int):
    """Retrieve or initialize cached inference engine for target scale.

    Args:
        scale (int): Upscaling scale factor (2, 4, or 8).

    Returns:
        Union[StudentInference, SwinIRInference]: Engine instance.
    """
    if scale not in _ENGINE_CACHE:
        logger.info(f"Initializing and caching inference engine for scale x{scale}...")
        _ENGINE_CACHE[scale] = select_model(scale=scale)
    return _ENGINE_CACHE[scale]


def process_image_enhancement(image_bytes: bytes, scale: int) -> bytes:
    """Process image bytes and return enhanced image bytes.

    Args:
        image_bytes (bytes): Raw uploaded input image bytes.
        scale (int): Desired output resolution scale (2, 4, or 8).

    Returns:
        bytes: Enhanced PNG image bytes.
    """
    if scale not in (2, 4, 8):
        raise ValueError(f"Invalid scale factor: {scale}. Supported values are 2, 4, and 8.")

    try:
        input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image file: {e}")

    logger.info(
        f"Processing backend image enhancement: input size {input_pil.size}, target scale x{scale}"
    )

    # Get cached inference engine
    engine = get_cached_engine(scale=scale)

    # Run inference pipeline
    output_pil = engine.enhance(input_pil)

    # Encode output image to PNG byte stream
    buffer = io.BytesIO()
    output_pil.save(buffer, format="PNG")
    output_bytes = buffer.getvalue()

    logger.info(
        f"Image enhancement complete: output size {output_pil.size}, encoded payload {len(output_bytes)} bytes"
    )

    return output_bytes
