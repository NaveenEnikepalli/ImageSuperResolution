"""Backend Configuration Module.

Author: Antigravity
Purpose: Centralized backend configuration settings for PixelLift API.
"""

from pathlib import Path
from typing import Dict, Any, List

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

# Upload Limits & Supported Types
MAX_UPLOAD_SIZE_MB: int = 10
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

SUPPORTED_FORMATS: List[str] = ["PNG", "JPG", "JPEG", "WEBP"]

# Scale Factors & Model Descriptions
AVAILABLE_SCALES: List[int] = [2, 4, 8]
DEFAULT_SCALE: int = 4

MODEL_METADATA: Dict[int, Dict[str, Any]] = {
    2: {
        "name": "Student x2",
        "type": "Lightweight RFDB Student",
        "scale": "2×",
        "parameters": "~186,603",
        "description": "Lightweight Student RFDB model optimized for fast 2x upscaling on low-resource devices.",
    },
    4: {
        "name": "Student x4",
        "type": "Lightweight RFDB Student",
        "scale": "4×",
        "parameters": "~436,011",
        "description": "Lightweight Student RFDB model for balanced 4x deep restoration and enhancement.",
    },
    8: {
        "name": "SwinIR x8",
        "type": "SwinIR-M Transformer",
        "scale": "8×",
        "parameters": "~11.8M",
        "description": "Official SwinIR-M Transformer engine for maximum 8x high-definition synthesis.",
    },
}
