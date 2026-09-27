"""Automatic Model Selector and Router for Super-Resolution Inference.

Author: Antigravity
Purpose: Clean model routing component selecting appropriate architecture based strictly on resolution scale:
  - scale = 2 -> Student x2 (StudentInference, scale=2)
  - scale = 4 -> Student x4 (StudentInference, scale=4)
  - scale = 8 -> SwinIR x8  (SwinIRInference, scale=8)

Raises explicit ValueError for unsupported scale values.
"""

import logging
from typing import Union, Optional, Dict, Any
import torch

from inference.student_inference import StudentInference
from inference.swinir_inference import SwinIRInference

logger = logging.getLogger(__name__)

# Supported resolution scales dictionary mapping
SUPPORTED_SCALES = (2, 4, 8)


def select_model(
    scale: int,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[str] = None,
) -> Union[StudentInference, SwinIRInference]:
    """Automatically select and instantiate super-resolution model engine based on requested scale.

    Routing table:
      scale=2 -> Student x2 engine
      scale=4 -> Student x4 engine
      scale=8 -> SwinIR x8 engine

    Args:
        scale (int): Requested upscaling scale factor (must be 2, 4, or 8).
        device (Optional[Union[str, torch.device]]): Target compute device (CPU/CUDA).
        checkpoint_path (Optional[str]): Optional custom checkpoint path override.

    Returns:
        Union[StudentInference, SwinIRInference]: Instantiated inference engine.

    Raises:
        ValueError: If requested scale is not in (2, 4, 8).
    """
    logger.info(f"Selecting model engine for resolution scale x{scale}...")

    if scale == 2:
        engine = StudentInference(scale=2, checkpoint_path=checkpoint_path, device=device)
    elif scale == 4:
        engine = StudentInference(scale=4, checkpoint_path=checkpoint_path, device=device)
    elif scale == 8:
        engine = SwinIRInference(scale=8, checkpoint_path=checkpoint_path, device=device)
    else:
        raise ValueError(
            f"Unsupported super-resolution scale factor x{scale}. "
            f"Supported resolution scales are 2x, 4x, and 8x."
        )

    return engine


# Backward compatibility helper alias
get_inference_engine = select_model
