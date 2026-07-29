"""Unit tests validating Validator, metrics wrappers, and AverageMeter components.

Author: Antigravity
Purpose: Test Validator correctness, parameters freezing, and interfaces.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from training.metrics.average_meter import AverageMeter
from training.metrics.psnr import PSNRMetric
from training.metrics.ssim import SSIMMetric
from training.models.student_model import StudentModel
from training.validator import Validator


# =====================================================================
# Mock Dataset
# =====================================================================

class DummyValDataset(Dataset):
    """Simple dummy dataset yielding LR/HR tensor pairs for validation tests."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # scale factor 4: LR (3, 16, 16) -> HR (3, 64, 64)
        # Create values between 0.0 and 1.0 to fit PIQ expectations
        return torch.rand(3, 16, 16), torch.rand(3, 64, 64)


# =====================================================================
# AverageMeter Tests
# =====================================================================

def test_average_meter() -> None:
    """Verify AverageMeter updates, aggregates, and resets correctly."""
    meter = AverageMeter()
    assert meter.average == 0.0

    meter.update(value=10.0, n=2)
    assert meter.average == 10.0

    meter.update(value=20.0, n=2)
    # Total sum: 10*2 + 20*2 = 60, total count: 4, average: 15.0
    assert meter.average == 15.0

    meter.reset()
    assert meter.average == 0.0


# =====================================================================
# Metrics Wrappers Tests
# =====================================================================

def test_psnr_metric_wrapper() -> None:
    """Verify PSNRMetric wrapper returns valid finite outputs."""
    psnr_fn = PSNRMetric()
    pred = torch.rand(1, 3, 32, 32)
    tgt = torch.rand(1, 3, 32, 32)

    score = psnr_fn(pred, tgt)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert not math.isinf(score)
    assert score > 0.0


def test_ssim_metric_wrapper() -> None:
    """Verify SSIMMetric wrapper returns valid finite outputs within [-1, 1]."""
    ssim_fn = SSIMMetric()
    pred = torch.rand(1, 3, 32, 32)
    tgt = torch.rand(1, 3, 32, 32)

    score = ssim_fn(pred, tgt)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert not math.isinf(score)
    assert -1.0 <= score <= 1.0


# =====================================================================
# Validator Engine Tests
# =====================================================================

def test_validator_init() -> None:
    """Verify Validator constructor binds dependencies correctly."""
    student = StudentModel(num_features=16, distilled_channels=8, num_blocks=2, scale=4)
    val_loader = DataLoader(DummyValDataset(4), batch_size=2)
    device = torch.device("cpu")
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()

    validator = Validator(
        student_model=student,
        validation_loader=val_loader,
        device=device,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric
    )

    assert validator.student_model == student
    assert validator.validation_loader == val_loader
    assert validator.device == device
    assert validator.psnr_metric == psnr_metric
    assert validator.ssim_metric == ssim_metric


def test_validator_execution_and_modes() -> None:
    """Verify Validator runs validation successfully without updating student weights."""
    device = torch.device("cpu")
    student = StudentModel(num_features=16, distilled_channels=8, num_blocks=2, scale=4)
    val_loader = DataLoader(DummyValDataset(4), batch_size=2)
    
    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()

    # Capture initial weights to verify they do not modify
    initial_weights = copy.deepcopy(next(student.parameters()).data)

    validator = Validator(
        student_model=student,
        validation_loader=val_loader,
        device=device,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric
    )

    # Force student to training mode first to verify Validator switches it
    student.train(True)
    assert student.training

    # Run evaluation
    results = validator.validate()

    # Assertions
    # A. Returned statistics format matches interface contract
    assert "psnr" in results
    assert "ssim" in results
    assert isinstance(results["psnr"], float)
    assert isinstance(results["ssim"], float)
    assert not math.isnan(results["psnr"])
    assert not math.isnan(results["ssim"])

    # B. Student was set to eval mode during/after execution
    assert not student.training, "Validator must set student model to eval mode!"

    # C. Student parameters remain unchanged (weights identical)
    updated_weights = next(student.parameters()).data
    assert torch.equal(initial_weights, updated_weights), (
        "Student weights must remain completely unchanged during validation!"
    )
