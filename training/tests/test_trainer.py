"""Unit tests validating Trainer components, parameters updates, and interfaces.

Author: Antigravity
Purpose: Test Trainer correctness, gradients, and statistics return formats.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from training.models.student_model import StudentModel
from training.teacher.swinir_wrapper import SwinIRWrapper
from training.losses.loss_manager import LossManager
from training.trainer import Trainer


# =====================================================================
# Mock Dataset
# =====================================================================

class DummyDataset(Dataset):
    """Simple dummy dataset generating matching LR/HR pairs for testing."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # scale factor 4: LR (3, 16, 16) -> HR (3, 64, 64)
        return torch.randn(3, 16, 16), torch.randn(3, 64, 64)


# =====================================================================
# Trainer Suite
# =====================================================================

def test_trainer_init() -> None:
    """Verify Trainer constructor binds dependencies correctly."""
    student = StudentModel(num_features=16, distilled_channels=8, num_blocks=2, scale=4)
    teacher = SwinIRWrapper(scale=4)
    loss_manager = LossManager()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    train_loader = DataLoader(DummyDataset(4), batch_size=2)
    device = torch.device("cpu")

    class MockConfig:
        epochs = 2

    trainer = Trainer(
        student_model=student,
        teacher_model=teacher,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        config=MockConfig(),
        device=device
    )

    assert trainer.student_model == student
    assert trainer.teacher_model == teacher
    assert trainer.loss_manager == loss_manager
    assert trainer.optimizer == optimizer
    assert trainer.scheduler == scheduler
    assert trainer.train_loader == train_loader
    assert trainer.device == device


def test_trainer_one_epoch_execution_and_updates() -> None:
    """Verify Trainer execution updates student weights, freezes teacher, and steps scheduler."""
    device = torch.device("cpu")
    
    # 1. Instantiate modules
    student = StudentModel(num_features=16, distilled_channels=8, num_blocks=2, scale=4)
    teacher = SwinIRWrapper(scale=4)
    loss_manager = LossManager(loss_config={"pixel_weight": 1.0, "kd_weight": 0.2})
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    train_loader = DataLoader(DummyDataset(4), batch_size=2)

    class MockConfig:
        class Training:
            epochs = 1
        training = Training()

    # Capture initial states to verify parameter changes/freezes
    initial_student_weights = copy.deepcopy(next(student.parameters()).data)
    initial_teacher_weights = copy.deepcopy(next(teacher.parameters()).data)
    initial_lr = optimizer.param_groups[0]["lr"]

    trainer = Trainer(
        student_model=student,
        teacher_model=teacher,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        config=MockConfig(),
        device=device
    )

    # 2. Run training for one epoch
    history = trainer.train()

    # 3. Assertions
    # A. Returned statistics format matches interface contract
    assert len(history) == 1
    stats = history[0]
    assert stats["epoch"] == 1
    assert "pixel_loss" in stats
    assert "kd_loss" in stats
    assert "total_loss" in stats
    assert isinstance(stats["pixel_loss"], float)
    assert isinstance(stats["kd_loss"], float)
    assert isinstance(stats["total_loss"], float)

    # B. Student parameters updated (weights changed)
    updated_student_weights = next(student.parameters()).data
    assert not torch.equal(initial_student_weights, updated_student_weights), (
        "Student parameters must update after optimization step!"
    )

    # C. Teacher parameters remain completely unchanged (weights identical)
    updated_teacher_weights = next(teacher.parameters()).data
    assert torch.equal(initial_teacher_weights, updated_teacher_weights), (
        "Teacher parameters must remain unchanged during distillation!"
    )

    # D. Teacher requires_grad remains False
    for param in teacher.parameters():
        assert not param.requires_grad

    # E. Scheduler steps correctly (lr shifts)
    updated_lr = optimizer.param_groups[0]["lr"]
    assert updated_lr != initial_lr, "Scheduler must step after completing epoch!"
