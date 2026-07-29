"""Unit tests validating CheckpointManager and DistillationLogger components.

Author: Antigravity
Purpose: Test CheckpointManager lifecycle, best model policies, and Logger outputs.
"""

import tempfile
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import copy

from training.logger import DistillationLogger
from training.utils.checkpoint import CheckpointManager
from training.models.student_model import StudentModel


# =====================================================================
# Logger Tests
# =====================================================================

def test_logger_console_and_file() -> None:
    """Verify DistillationLogger logs statistics to file if configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        dist_logger = DistillationLogger(log_dir=log_dir)

        # Log an epoch stats
        dist_logger.log_epoch(
            epoch=1,
            train_stats={"pixel_loss": 0.05, "kd_loss": 0.02, "total_loss": 0.054},
            val_stats={"psnr": 28.5, "ssim": 0.82},
            lr=1e-4
        )

        log_file = log_dir / "log.txt"
        assert log_file.exists(), "Logger must create log.txt in log_dir!"

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Epoch: 1 | LR: 0.000100" in content
        assert "Train Pixel Loss: 0.050000" in content
        assert "Val PSNR: 28.5000 dB | SSIM: 0.8200" in content


# =====================================================================
# CheckpointManager Tests
# =====================================================================

def test_checkpoint_manager_best_policy_and_restoration() -> None:
    """Verify CheckpointManager saves, overwrites, filters best PSNR, and restores states."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        
        # Instantiate dependencies
        student = StudentModel(num_features=16, distilled_channels=8, num_blocks=2, scale=4)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            student_model=student,
            optimizer=optimizer,
            scheduler=scheduler
        )

        latest_path = checkpoint_dir / "latest.pth"
        best_path = checkpoint_dir / "best.pth"

        # Check initialized state
        assert manager.best_psnr == 0.0

        # Capture initial parameter states
        initial_student_weights = copy.deepcopy(next(student.parameters()).data)

        # -----------------------------------------------------------------
        # 1. Save epoch 1: PSNR = 28.0 (This is the new best)
        # -----------------------------------------------------------------
        manager.save(
            epoch=1,
            training_history=[{"epoch": 1, "total_loss": 0.05}],
            validation_metrics={"psnr": 28.0, "ssim": 0.8}
        )

        assert latest_path.exists()
        assert best_path.exists()
        assert manager.best_psnr == 28.0

        # Load epoch 1 best checkpoint details and verify
        ckpt_data = torch.load(best_path, map_location="cpu")
        assert ckpt_data["epoch"] == 1
        assert ckpt_data["best_psnr"] == 28.0

        # -----------------------------------------------------------------
        # 2. Save epoch 2: PSNR = 27.5 (Lower, latest.pth updates but best.pth does not)
        # -----------------------------------------------------------------
        # Modify student weights to represent training progress
        with torch.no_grad():
            next(student.parameters()).data.add_(1.0)
        epoch_2_student_weights = copy.deepcopy(next(student.parameters()).data)

        manager.save(
            epoch=2,
            training_history=[{"epoch": 1, "total_loss": 0.05}, {"epoch": 2, "total_loss": 0.04}],
            validation_metrics={"psnr": 27.5, "ssim": 0.79}
        )

        assert manager.best_psnr == 28.0  # best_psnr does not decrease
        
        # Verify latest.pth has epoch 2 details
        latest_data = torch.load(latest_path, map_location="cpu")
        assert latest_data["epoch"] == 2
        assert latest_data["best_psnr"] == 28.0
        
        # Verify best.pth is NOT overwritten and still holds epoch 1 details
        best_data = torch.load(best_path, map_location="cpu")
        assert best_data["epoch"] == 1
        assert best_data["validation_metrics"]["psnr"] == 28.0

        # -----------------------------------------------------------------
        # 3. Save epoch 3: PSNR = 29.0 (New best, updates both files)
        # -----------------------------------------------------------------
        with torch.no_grad():
            next(student.parameters()).data.add_(1.0)
        epoch_3_student_weights = copy.deepcopy(next(student.parameters()).data)

        manager.save(
            epoch=3,
            training_history=[
                {"epoch": 1, "total_loss": 0.05},
                {"epoch": 2, "total_loss": 0.04},
                {"epoch": 3, "total_loss": 0.03}
            ],
            validation_metrics={"psnr": 29.0, "ssim": 0.82}
        )

        assert manager.best_psnr == 29.0

        # Verify best.pth now has epoch 3 details
        best_data = torch.load(best_path, map_location="cpu")
        assert best_data["epoch"] == 3
        assert best_data["best_psnr"] == 29.0
        assert best_data["validation_metrics"]["psnr"] == 29.0

        # -----------------------------------------------------------------
        # 4. Test loading / restoration workflow
        # -----------------------------------------------------------------
        # Set student weights to a clean layout to verify restoration
        with torch.no_grad():
            next(student.parameters()).copy_(initial_student_weights)

        assert torch.equal(next(student.parameters()).data, initial_student_weights)

        # Restore from epoch 3 (latest.pth)
        meta = manager.load(latest_path)

        assert meta["epoch"] == 3
        assert meta["best_psnr"] == 29.0
        assert len(meta["training_history"]) == 3
        assert meta["validation_metrics"]["psnr"] == 29.0

        # Verify weights were restored to epoch 3 values
        restored_student_weights = next(student.parameters()).data
        assert torch.equal(restored_student_weights, epoch_3_student_weights)
        assert not torch.equal(restored_student_weights, initial_student_weights)
