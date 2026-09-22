"""Unit and integration tests validating SwinIR teacher backbone, SwinIRWrapper, and distillation workflow.

Author: Antigravity
Purpose: Comprehensive verification of SwinIR teacher construction, window-size padding,
         invariants, strict checkpoint validation error handling, pretrained weight loading,
         and end-to-end distillation integration.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from training.teacher.swinir_model import SwinIR
from training.teacher.swinir_wrapper import SwinIRWrapper
from training.models.student_model import StudentModel
from training.losses.loss_manager import LossManager


def test_swinir_model_construction() -> None:
    """Verify SwinIR model instantiates with expected Classical SR x4 structural components."""
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    assert isinstance(model, nn.Module)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 11899839, f"Parameter count mismatch: expected 11899839, got {param_count}"


def test_swinir_wrapper_invariants() -> None:
    """Verify SwinIRWrapper enforces parameter freezing and eval mode persistence."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)

    assert teacher.scale == 4
    assert teacher.in_channels == 3
    assert not teacher.training, "Teacher must be initialized in evaluation mode"

    # Verify all teacher parameters are frozen
    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher parameter {name} was not frozen!"

    # Verify train(True) override prevents entering training mode
    teacher.train(True)
    assert not teacher.training, "Teacher must stay in eval mode even after train(True) call!"


def test_swinir_forward_aligned_shape() -> None:
    """Verify SwinIRWrapper forward execution maps 48x48 LR input to 192x192 HR output."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)

    # Input tensor with requires_grad to verify no_grad context
    dummy_input = torch.randn(2, 3, 48, 48, requires_grad=True)
    output = teacher(dummy_input)

    assert list(output.shape) == [2, 3, 192, 192]
    assert output.dtype == dummy_input.dtype
    assert output.device == dummy_input.device

    # Verify output does not track gradients
    assert output.grad_fn is None, "Teacher forward pass must run in torch.no_grad()"

    # Verify finite output (no NaN or Inf values)
    assert torch.isfinite(output).all(), "Teacher output contains NaN or Inf values"


def test_swinir_forward_non_aligned_shape() -> None:
    """Verify SwinIRWrapper window padding handles arbitrary spatial dimensions (45x47 -> 180x188)."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)

    dummy_input = torch.randn(2, 3, 45, 47)
    output = teacher(dummy_input)

    # 45 * 4 = 180, 47 * 4 = 188
    assert list(output.shape) == [2, 3, 180, 188]
    assert torch.isfinite(output).all(), "Non-aligned output contains NaN or Inf values"


def test_swinir_load_checkpoint_missing_file() -> None:
    """Verify load_checkpoint raises FileNotFoundError if checkpoint path does not exist."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)

    with pytest.raises(FileNotFoundError, match="Teacher checkpoint file not found"):
        teacher.load_checkpoint("non_existent_weights.pth")


def test_swinir_load_checkpoint_incompatible_structure() -> None:
    """Verify load_checkpoint raises RuntimeError if checkpoint contains incompatible state_dict."""
    teacher = SwinIRWrapper(scale=4, in_channels=3)

    # Create dummy incompatible checkpoint
    incompatible_dict = {"conv_first.weight": torch.randn(1, 1, 1, 1)}

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        torch.save(incompatible_dict, tmp_path)

    try:
        with pytest.raises(RuntimeError, match="Incompatible SwinIR teacher checkpoint"):
            teacher.load_checkpoint(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_swinir_load_real_checkpoint_strict() -> None:
    """Verify load_checkpoint performs 100% strict verification on official SwinIR-M pretrained weights."""
    ckpt_path = Path("checkpoints/teacher/swinir_x4.pth")
    if not ckpt_path.exists():
        pytest.skip("Pretrained SwinIR checkpoint file not found at checkpoints/teacher/swinir_x4.pth")

    teacher = SwinIRWrapper(scale=4, in_channels=3)
    teacher.load_checkpoint(ckpt_path)

    assert teacher.is_loaded is True
    assert sum(p.numel() for p in teacher.parameters() if p.requires_grad) == 0
    assert not teacher.training

    # Execute forward pass with loaded weights
    dummy_input = torch.randn(2, 3, 48, 48)
    output = teacher(dummy_input)
    assert list(output.shape) == [2, 3, 192, 192]
    assert torch.isfinite(output).all()


def test_student_teacher_distillation_integration() -> None:
    """Integration test verifying gradient flow during student-teacher distillation.

    Verifies:
    1. Student forward & teacher forward execute without shape mismatches.
    2. LossManager aggregates pixel loss + KD loss into total_loss.
    3. total_loss.backward() updates student gradients.
    4. Teacher parameters receive NO gradients.
    """
    device = torch.device("cpu")

    student = StudentModel(in_channels=3, out_channels=3, num_features=48, distilled_channels=24, num_blocks=3, scale=4).to(device)
    teacher = SwinIRWrapper(scale=4, in_channels=3).to(device)
    loss_manager = LossManager(loss_config={"pixel_weight": 1.0, "kd_weight": 0.1}).to(device)

    # Dummy inputs
    lr_batch = torch.randn(2, 3, 48, 48, device=device)
    gt_hr_batch = torch.randn(2, 3, 192, 192, device=device)

    # Student forward
    student_sr = student(lr_batch)
    assert list(student_sr.shape) == [2, 3, 192, 192]

    # Teacher forward
    teacher_sr = teacher(lr_batch)
    assert list(teacher_sr.shape) == [2, 3, 192, 192]
    assert teacher_sr.grad_fn is None

    # Loss computation
    loss_dict = loss_manager(student_sr, teacher_sr, gt_hr_batch)
    total_loss = loss_dict["total_loss"]

    assert total_loss.requires_grad is True

    # Backpropagation
    student.zero_grad()
    total_loss.backward()

    # Verify student parameters received non-zero gradients
    student_grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert len(student_grads) > 0
    assert any(g is not None and torch.any(g != 0) for g in student_grads)

    # Verify teacher parameters received NO gradients
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter {name} unexpectedly received gradients!"
