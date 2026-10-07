"""Backend Enhancement Service.

Author: Antigravity
Purpose: High-level business logic orchestrating image validation, model routing,
inference timing, and output encoding.
"""

import time
import logging
from typing import Dict, Any, Optional
from PIL import Image

try:
    from backend.app.config import MODEL_METADATA
    from backend.app.services.image_service import (
        validate_image_file,
        load_and_preprocess_image,
        encode_image_to_png,
        get_image_info,
    )
    from backend.app.services.model_service import get_model_engine
except ModuleNotFoundError:
    from app.config import MODEL_METADATA
    from app.services.image_service import (
        validate_image_file,
        load_and_preprocess_image,
        encode_image_to_png,
        get_image_info,
    )
    from app.services.model_service import get_model_engine

logger = logging.getLogger(__name__)


def enhance_image(
    file_bytes: bytes,
    scale: int = 4,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute image enhancement pipeline."""
    # Validate input image payload
    is_valid, err_msg = validate_image_file(file_bytes, filename=filename)
    if not is_valid:
        raise ValueError(err_msg or "Invalid image file.")

    if scale not in (2, 4, 8):
        raise ValueError(f"Unsupported scale factor x{scale}. Supported scales are 2, 4, and 8.")

    # Preprocess image
    input_pil = load_and_preprocess_image(file_bytes)
    in_w, in_h = input_pil.size
    input_info = get_image_info(input_pil, len(file_bytes))

    logger.info(
        f"Processing image enhancement: '{filename or 'input_image'}' ({in_w}x{in_h}) at scale x{scale}"
    )

    # Fetch cached model engine
    engine = get_model_engine(scale=scale)
    model_info = MODEL_METADATA.get(scale, {"name": f"Model x{scale}", "type": "Super Resolution"})

    # Measure inference execution time
    start_time = time.perf_counter()
    output_pil = engine.enhance(input_pil)
    elapsed_time = time.perf_counter() - start_time

    out_w, out_h = output_pil.size

    # Encode output to PNG
    output_bytes = encode_image_to_png(output_pil)
    output_info = get_image_info(output_pil, len(output_bytes))

    logger.info(
        f"Enhancement complete in {elapsed_time:.3f}s: ({in_w}x{in_h}) -> ({out_w}x{out_h})"
    )

    return {
        "success": True,
        "filename": filename or "image.png",
        "input_image": input_pil,
        "output_image": output_pil,
        "output_bytes": output_bytes,
        "input_size": (in_w, in_h),
        "output_size": (out_w, out_h),
        "scale": scale,
        "model_name": model_info["name"],
        "model_metadata": model_info,
        "inference_time": elapsed_time,
        "input_info": input_info,
        "output_info": output_info,
    }
