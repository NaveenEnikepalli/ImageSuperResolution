"""Unit tests validating Reconstruction imports, upsampling scale dimensions, and shapes mapping.

Author: Antigravity
Purpose: Unit tests for Reconstruction.
"""

import pytest
import torch
from training.models.modules.reconstruction import Reconstruction


def test_reconstruction_init() -> None:
    """Verify Reconstruction properties initialization."""
    recon = Reconstruction(num_features=32, out_channels=3, scale=4)
    assert recon.num_features == 32
    assert recon.out_channels == 3
    assert recon.scale == 4
    assert isinstance(recon, torch.nn.Module)


def test_reconstruction_forward() -> None:
    """Verify Reconstruction scales output height and width by the scale factor."""
    recon = Reconstruction(num_features=32, out_channels=3, scale=4)
    dummy_input = torch.randn(2, 32, 48, 48)
    
    with torch.no_grad():
        output = recon(dummy_input)
        
    assert list(output.shape) == [2, 3, 192, 192]  # Scale x4: 48 * 4 = 192
    assert output.dtype == dummy_input.dtype
    assert output.device == dummy_input.device


def test_reconstruction_channels_mismatch() -> None:
    """Verify Reconstruction validates input feature map channels."""
    recon = Reconstruction(num_features=32, out_channels=3, scale=4)
    invalid_input = torch.randn(2, 16, 48, 48)
    
    with pytest.raises(AssertionError, match="Input channels mismatch"):
        recon(invalid_input)
