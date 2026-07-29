"""Checkpoint lifecycle manager for Super-Resolution Knowledge Distillation.

Author: Antigravity
Purpose: Save and load latest and best models, optimizers, and schedulers states.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Lifecycle manager for model, optimizer, and scheduler checkpoints.

    Exposes save() and load() interfaces. It determines whether validation
    metrics qualify for saving a new best checkpoint, and handles state loading.

    Example:
        mgr = CheckpointManager(
            checkpoint_dir="outputs/runs/checkpoints",
            student_model=student,
            optimizer=opt,
            scheduler=sched
        )
        mgr.save(epoch=1, training_history=history, validation_metrics={"psnr": 29.5, "ssim": 0.85})
        meta = mgr.load("outputs/runs/checkpoints/latest.pth")
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        student_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir (Union[str, Path]): Path where checkpoints are stored.
            student_model (nn.Module): Student model to checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer to checkpoint.
            scheduler (Optional[Any]): Optional scheduler to checkpoint. Defaults to None.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.student_model = student_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_psnr = 0.0

        # Create output directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        training_history: List[Dict[str, Any]],
        validation_metrics: Dict[str, float],
    ) -> None:
        """Serialize current state parameters and save to disk.

        Saves to latest.pth. If validation PSNR exceeds current best, saves to best.pth
        as well.

        Args:
            epoch (int): The current training epoch.
            training_history (List[Dict[str, Any]]): Training stats accumulated so far.
            validation_metrics (Dict[str, float]): Validation metrics for the current epoch.
        """
        current_psnr = validation_metrics.get("psnr", 0.0)

        # Check if the current validation metric outperforms the best PSNR
        is_best = current_psnr > self.best_psnr
        if is_best:
            self.best_psnr = current_psnr

        # Construct checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "best_psnr": self.best_psnr,
            "student_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_history": training_history,
            "validation_metrics": validation_metrics,
        }

        # Save to latest.pth
        latest_path = self.checkpoint_dir / "latest.pth"
        try:
            torch.save(checkpoint, latest_path)
            logger.info(f"Successfully saved latest checkpoint: {latest_path}")
        except Exception as e:
            logger.error(f"Failed to save latest checkpoint. Error: {e}")

        # Save to best.pth if it is the best performing epoch so far
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            try:
                torch.save(checkpoint, best_path)
                logger.info(f"New best checkpoint saved with PSNR: {self.best_psnr:.4f} dB at {best_path}")
            except Exception as e:
                logger.error(f"Failed to save best checkpoint. Error: {e}")

    def load(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load state parameters from a serialized checkpoint on disk.

        Restores state of student model, optimizer, scheduler, and updates best_psnr.

        Args:
            checkpoint_path (Union[str, Path]): Path to target checkpoint file.

        Returns:
            Dict[str, Any]: Restored metadata parameters dict containing:
                - "epoch": restored epoch integer.
                - "best_psnr": restored best PSNR score.
                - "training_history": training history list.
                - "validation_metrics": validation metrics dict.
        """
        path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint parameters from: {path}")

        # Load file onto CPU memory mapping to avoid cuda mismatches
        checkpoint = torch.load(path, map_location="cpu")

        # Restore state dicts
        self.student_model.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore best PSNR tracker
        self.best_psnr = checkpoint.get("best_psnr", 0.0)

        logger.info(
            f"Successfully restored state from epoch {checkpoint['epoch']} "
            f"(best PSNR recorded: {self.best_psnr:.4f} dB)"
        )

        return {
            "epoch": checkpoint["epoch"],
            "best_psnr": self.best_psnr,
            "training_history": checkpoint["training_history"],
            "validation_metrics": checkpoint["validation_metrics"],
        }

    @staticmethod
    def save_checkpoint(state_dict: Dict[str, Any], checkpoint_path: Union[str, Path]) -> None:
        """Legacy helper to save raw state dictionaries to a path.

        Args:
            state_dict (Dict[str, Any]): Raw dict states.
            checkpoint_path (Union[str, Path]): Target filepath destination.
        """
        dest_path = Path(checkpoint_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, dest_path)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
        """Legacy helper to find latest saved checkpoint in a directory.

        Args:
            checkpoint_dir (Union[str, Path]): Checkpoint directory.

        Returns:
            Optional[Path]: Latest checkpoint filepath if found.
        """
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists():
            return None
        pth_files = list(dir_path.glob("*.pth"))
        if not pth_files:
            return None
        # Return path with maximum modification time
        return max(pth_files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def resume_training(checkpoint_dir: Union[str, Path]) -> tuple[Optional[Dict[str, Any]], int]:
        """Legacy helper to reload training states from latest checkpoint.

        Args:
            checkpoint_dir (Union[str, Path]): Checkpoint directory.

        Returns:
            tuple[Optional[Dict[str, Any]], int]: Tuple containing the loaded state dict
                and the next starting epoch number.
        """
        latest = CheckpointManager.get_latest_checkpoint(checkpoint_dir)
        if latest is None:
            return None, 1
        state = torch.load(latest, map_location="cpu")
        start_epoch = state.get("epoch", 0) + 1
        return state, start_epoch

