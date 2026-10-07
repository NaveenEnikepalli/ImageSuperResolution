"""Backend Image Processing Service.

Author: Antigravity
Purpose: Decoupled image validation, RGB mode normalization, metadata extraction,
and PNG byte stream encoding for backend processing.
"""

import io
import logging
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageOps

try:
    from backend.app.config import (
        MAX_UPLOAD_SIZE_BYTES,
        MAX_UPLOAD_SIZE_MB,
        SUPPORTED_FORMATS,
    )
except ModuleNotFoundError:
    from app.config import (
        MAX_UPLOAD_SIZE_BYTES,
        MAX_UPLOAD_SIZE_MB,
        SUPPORTED_FORMATS,
    )

logger = logging.getLogger(__name__)


def validate_image_file(
    file_bytes: bytes, filename: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Validate image bytes for non-emptiness, maximum size limits, and valid formats."""
    if not file_bytes or len(file_bytes) == 0:
        return False, "Uploaded file payload is empty."

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return (
            False,
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img_format = (img.format or "").upper()
            if img_format not in SUPPORTED_FORMATS and img_format not in ["JPEG", "MPO"]:
                return (
                    False,
                    f"Unsupported image format '{img_format}'. Supported formats are: {', '.join(SUPPORTED_FORMATS)}.",
                )
            img.verify()
    except Exception as e:
        logger.error(f"Image decoding failure for file {filename}: {e}")
        return False, f"Invalid or corrupted image file: {str(e)}"

    return True, None


def load_and_preprocess_image(file_bytes: bytes) -> Image.Image:
    """Decode raw image bytes and convert to PIL Image in RGB mode."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def encode_image_to_png(image: Image.Image) -> bytes:
    """Encode PIL Image object to high-quality PNG byte stream."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def get_image_info(image: Image.Image, file_size_bytes: int) -> Dict[str, Any]:
    """Extract metadata details from PIL Image."""
    w, h = image.size
    size_str = (
        f"{file_size_bytes / 1024:.1f} KB"
        if file_size_bytes < 1024 * 1024
        else f"{file_size_bytes / (1024 * 1024):.2f} MB"
    )
    return {
        "width": w,
        "height": h,
        "resolution": f"{w} × {h}",
        "file_size": size_str,
        "mode": image.mode,
    }
