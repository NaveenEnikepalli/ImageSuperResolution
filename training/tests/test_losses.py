"""Unit tests validating SwinIRWrapper teacher model and modular Loss Framework.

Author: Antigravity
Purpose: Test code correctness for SwinIRWrapper, PixelLoss, KnowledgeDistillationLoss, and LossManager.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from training.teacher.swinir_wrapper import SwinIRWrapper
from training.losses.pixel_loss import PixelLoss
from training.losses.kd_loss import KnowledgeDistillationLoss
from training.losses.loss_manager import LossManager
from training.utils.config import ExperimentConfig, ModelConfig


# =====================================================================
# SwinIRWrapper Tests
# =====================================================================

def test_teacher_wrapper_init() -> None:
    """Verify SwinIRWrapper initializes in eval mode and freezes all parameters."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)
    assert teacher.scale == 4
    assert teacher.in_channels == 3
    assert not teacher.training, "Teacher must be initialized in evaluation mode!"

    # Verify that all parameters have requires_grad set to False
    for param in teacher.parameters():
        assert not param.requires_grad, "Teacher parameters must be frozen!"


def test_teacher_wrapper_train_override() -> None:
    """Verify SwinIRWrapper stays in evaluation mode even if train() is called."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)
    teacher.train(True)
    assert not teacher.training, "Teacher must stay in eval mode even after train(True)!"


def test_teacher_wrapper_forward() -> None:
    """Verify forward execution maps low-res shapes to upscaled high-res shapes."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)
    
    # Input has gradients enabled to verify no_grad block drops graph tracking
    dummy_input = torch.randn(2, 3, 16, 16, requires_grad=True)
    output = teacher(dummy_input)

    assert list(output.shape) == [2, 3, 64, 64]
    assert output.dtype == dummy_input.dtype
    assert output.device == dummy_input.device
    
    # Verify that the output does not track gradients (since forward runs in no_grad context)
    assert output.grad_fn is None, "Teacher forward must execute in torch.no_grad()!"


def test_teacher_wrapper_load_checkpoint_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Verify load_checkpoint handles non-existent paths gracefully without crashing."""
    teacher = SwinIRWrapper(scale=2, in_channels=3)
    
    # Attempting to load from a non-existent path
    teacher.load_checkpoint("non_existent_weights.pth")
    assert not teacher.is_loaded
    
    # Parameters must remain frozen and eval mode kept
    for param in teacher.parameters():
        assert not param.requires_grad
    assert not teacher.training


# =====================================================================
# Individual Losses Tests
# =====================================================================

def test_pixel_loss() -> None:
    """Verify PixelLoss calculates raw unweighted L1 loss correctly."""
    loss_fn = PixelLoss()
    
    student_sr = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt_hr = torch.tensor([[[[1.5, 1.5], [3.5, 3.5]]]])
    
    loss = loss_fn(student_sr, gt_hr)
    # L1: (|1.0-1.5| + |2.0-1.5| + |3.0-3.5| + |4.0-3.5|) / 4 = (0.5+0.5+0.5+0.5)/4 = 0.5
    assert torch.allclose(loss, torch.tensor(0.5))


def test_kd_loss() -> None:
    """Verify KnowledgeDistillationLoss calculates raw unweighted L1 loss correctly."""
    loss_fn = KnowledgeDistillationLoss()
    
    student_sr = torch.tensor([[[[2.0, 3.0]]]])
    teacher_sr = torch.tensor([[[[4.0, 3.0]]]])
    
    loss = loss_fn(student_sr, teacher_sr)
    # L1: (|2.0-4.0| + |3.0-3.0|) / 2 = 1.0
    assert torch.allclose(loss, torch.tensor(1.0))


# =====================================================================
# LossManager Tests
# =====================================================================

def test_loss_manager_default_weights() -> None:
    """Verify LossManager default weights configuration."""
    manager = LossManager()
    assert manager.pixel_weight == 1.0
    assert manager.kd_weight == 0.2


def test_loss_manager_dict_weights() -> None:
    """Verify LossManager correctly parses weights from a configuration dictionary."""
    config_dict = {
        "pixel_weight": 0.8,
        "kd_weight": 0.5
    }
    manager = LossManager(loss_config=config_dict)
    assert manager.pixel_weight == 0.8
    assert manager.kd_weight == 0.5


def test_loss_manager_config_object_weights() -> None:
    """Verify LossManager correctly parses weights from a configuration object hierarchy."""
    # Build ExperimentConfig mock
    mock_config = ExperimentConfig(
        model=ModelConfig(
            type="student",
            name="edsr_small"
        )
    )
    
    # 1. Custom distillation weights configuration on ExperimentConfig
    mock_config.losses = {
        "distillation": {
            "alpha_pixel": 0.75,
            "beta_feature": 0.15
        }
    }
    
    manager = LossManager(loss_config=mock_config)
    assert manager.pixel_weight == 0.75
    assert manager.kd_weight == 0.15


def test_loss_manager_forward_interface() -> None:
    """Verify LossManager forward output dictionary interface and loss combination math."""
    manager = LossManager(loss_config={"pixel_weight": 2.0, "kd_weight": 0.5})
    
    student_sr = torch.tensor([[[[1.0, 2.0]]]])
    teacher_sr = torch.tensor([[[[3.0, 2.0]]]])  # KD L1 = 1.0
    gt_hr = torch.tensor([[[[1.0, 4.0]]]])       # Pixel L1 = 1.0
    
    loss_dict = manager(student_sr, teacher_sr, gt_hr)
    
    # Verify exact keys presence
    assert "pixel_loss" in loss_dict
    assert "kd_loss" in loss_dict
    assert "total_loss" in loss_dict
    
    # Verify raw losses are returned unweighted in sub-keys
    assert torch.allclose(loss_dict["pixel_loss"], torch.tensor(1.0))
    assert torch.allclose(loss_dict["kd_loss"], torch.tensor(1.0))
    
    # Verify combination math: total = 2.0 * 1.0 + 0.5 * 1.0 = 2.5
    expected_total = 2.0 * 1.0 + 0.5 * 1.0
    assert torch.allclose(loss_dict["total_loss"], torch.tensor(expected_total))
