"""Backend Model Service Module.

Author: Antigravity
Purpose: Thread-safe, persistent in-memory caching for super-resolution model engines.
Loads trained weights from disk once per scale and reuses cached model instances.
"""

import logging
from typing import Dict, Any, Union, Optional
import torch

from inference.model_selector import select_model

logger = logging.getLogger(__name__)

# Persistent in-memory model cache dictionary
_MODEL_CACHE: Dict[int, Any] = {}


def get_model_engine(
    scale: int, device: Optional[Union[str, torch.device]] = None
) -> Any:
    """Retrieve or instantiate cached super-resolution model engine for target scale.

    Routing:
      scale == 2 -> StudentInference(scale=2) (Student x2)
      scale == 4 -> StudentInference(scale=4) (Student x4)
      scale == 8 -> SwinIRInference(scale=8)  (SwinIR x8)

    Args:
        scale (int): Requested scale factor (2, 4, or 8).
        device (Optional[Union[str, torch.device]]): Target compute device.

    Returns:
        Any: Cached inference engine instance.

    Raises:
        ValueError: If scale is not supported (not 2, 4, or 8).
    """
    if scale not in (2, 4, 8):
        raise ValueError(
            f"Unsupported scale factor x{scale}. Supported scales are 2, 4, and 8."
        )

    cache_key = scale
    if cache_key not in _MODEL_CACHE:
        logger.info(f"Loading and caching model engine for scale x{scale}...")
        engine = select_model(scale=scale, device=device)
        _MODEL_CACHE[cache_key] = engine
        logger.info(f"Successfully cached model engine for scale x{scale}.")

    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Clear all cached model engine instances from memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Cleared model cache.")
