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


from app.services.enhancement_service import enhance_image


def process_image_enhancement(image_bytes: bytes, scale: int) -> bytes:
    """Process image bytes and return enhanced PNG image bytes via enhancement service.

    Args:
        image_bytes (bytes): Raw uploaded input image bytes.
        scale (int): Desired output resolution scale (2, 4, or 8).

    Returns:
        bytes: Enhanced PNG image bytes.
    """
    result = enhance_image(file_bytes=image_bytes, scale=scale)
    return result["output_bytes"]

