"""Unit tests validating StudentModel assembly, configuration system loading, and modular forwarding logic.

Author: Antigravity
Purpose: Unit tests for StudentModel.
"""

# pyrefly: ignore [missing-import]
import pytest
import torch
from training.models.student_model import StudentModel
from training.utils.config import ExperimentConfig, ModelConfig


def test_student_model_default_init() -> None:
    """Verify StudentModel default parameters loading."""
    model = StudentModel()
    assert model.scale == 4
    assert model.num_blocks == 3
    assert model.num_features == 48
    assert model.distilled_channels == 24
    assert model.in_channels == 3
    assert model.out_channels == 3
    assert isinstance(model, torch.nn.Module)


def test_student_model_config_init() -> None:
    """Verify StudentModel successfully extracts parameters from the config module explicitly."""
    # Build custom mock config
    mock_config = ExperimentConfig(
        experiment_name="test_student_run",
        model=ModelConfig(
            type="student",
            name="edsr_small",
            scale=2,
            num_blocks=4,
            num_channels=16
        )
    )

    # Instantiate using explicit constructor parameters extracted from mock_config
    model = StudentModel(
        scale=mock_config.model.scale,
        num_blocks=mock_config.model.num_blocks,
        num_features=mock_config.model.num_channels,
    )
    assert model.scale == 2
    assert model.num_blocks == 4
    assert model.num_features == 16


def test_student_model_forward() -> None:
    """Verify StudentModel coordinates forward pass returning expected output scale shapes."""
    model = StudentModel()
    dummy_input = torch.randn(2, 3, 48, 48)  # Batch 2, RGB, 48x48
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert list(output.shape) == [2, 3, 192, 192]  # scale=4: 48 * 4 = 192
    assert output.dtype == dummy_input.dtype
    assert output.device == dummy_input.device


def test_student_model_diagnostics() -> None:
    """Verify StudentModel integration with model utility diagnostic checkers."""
    from training.models.utils.model_utils import (
        count_parameters,
        estimate_model_size,
        verify_device,
        verify_dtype
    )
    
    model = StudentModel()
    params_count = count_parameters(model)
    assert params_count > 0

    size_mb = estimate_model_size(model)
    assert size_mb >= 0.0

    assert verify_device(model, torch.device("cpu"))
    assert verify_dtype(model, torch.float32)
