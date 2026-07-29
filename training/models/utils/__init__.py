"""
Package initializer for models/utils, exporting parameter diagnostic and weights initialization utilities.
"""

from training.models.utils.model_utils import (
    count_parameters,
    count_trainable_parameters,
    estimate_model_size,
    print_model_summary,
    verify_forward_signature,
    verify_device,
    verify_dtype,
    validate_tensor
)
from training.models.utils.initialization import (
    initialize_kaiming_normal,
    initialize_xavier_normal,
    initialize_normal,
    initialize_constant,
    custom_initialize
)

__all__ = [
    "count_parameters",
    "count_trainable_parameters",
    "estimate_model_size",
    "print_model_summary",
    "verify_forward_signature",
    "verify_device",
    "verify_dtype",
    "validate_tensor",
    "initialize_kaiming_normal",
    "initialize_xavier_normal",
    "initialize_normal",
    "initialize_constant",
    "custom_initialize"
]
