"""Unit tests validating GlobalFeatureFusion imports, initialization, and list-based forward signature.

Author: Antigravity
Purpose: Unit tests for GlobalFeatureFusion.
"""

import pytest
import torch
from training.models.modules.global_feature_fusion import GlobalFeatureFusion


def test_feature_fusion_init() -> None:
    """Verify GlobalFeatureFusion instantiation properties."""
    fusion = GlobalFeatureFusion(num_blocks=6, num_features=32)
    assert fusion.num_blocks == 6
    assert fusion.num_features == 32
    assert isinstance(fusion, torch.nn.Module)


def test_feature_fusion_forward() -> None:
    """Verify GlobalFeatureFusion aggregates inputs list correctly."""
    fusion = GlobalFeatureFusion(num_blocks=4, num_features=32)
    dummy_list = [torch.randn(2, 32, 48, 48) for _ in range(4)]
    
    with torch.no_grad():
        output = fusion(dummy_list)
        
    assert list(output.shape) == [2, 32, 48, 48]
    assert output.dtype == dummy_list[0].dtype
    assert output.device == dummy_list[0].device


def test_feature_fusion_list_length_mismatch() -> None:
    """Verify GlobalFeatureFusion raises error if input list count is wrong."""
    fusion = GlobalFeatureFusion(num_blocks=4, num_features=32)
    invalid_list = [torch.randn(2, 32, 48, 48) for _ in range(3)]  # 3 items instead of 4
    
    with pytest.raises(AssertionError, match="Input list size mismatch"):
        fusion(invalid_list)


def test_feature_fusion_channels_mismatch() -> None:
    """Verify GlobalFeatureFusion raises error if inner list elements have mismatched channels."""
    fusion = GlobalFeatureFusion(num_blocks=2, num_features=32)
    invalid_list = [
        torch.randn(2, 32, 48, 48),
        torch.randn(2, 16, 48, 48)  # Mismatched channel count (16)
    ]
    
    with pytest.raises(AssertionError, match="Feature channels mismatch at index 1"):
        fusion(invalid_list)
