"""Trainer orchestration engine for Super-Resolution Knowledge Distillation.

Author: Antigravity
Purpose: Coordinates training epochs, model execution, backpropagation, and stats aggregation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer engine orchestrating student optimization and teacher distillation.

    Constructor receives all components through dependency injection. It coordinates
    epoch and batch loops, device castings, forward-backward pipelines, optimization steps,
    and returns aggregated epoch-wise loss statistics.

    Example:
        trainer = Trainer(
            student_model=student,
            teacher_model=teacher,
            loss_manager=loss_mgr,
            optimizer=opt,
            scheduler=sched,
            train_loader=loader,
            config=config,
            device=device,
            validator=validator,
            logger=dist_logger,
            checkpoint_manager=ckpt_mgr
        )
        stats = trainer.train()
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        loss_manager: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        config: Any,
        device: torch.device,
        validator: Optional[Any] = None,
        logger: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            student_model (nn.Module): Student Super-Resolution model to train.
            teacher_model (nn.Module): Pretrained Teacher model (SwinIRWrapper).
            loss_manager (nn.Module): Extensible LossManager combining weighted losses.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            scheduler (Any): PyTorch learning rate scheduler.
            train_loader (DataLoader): Training data loader.
            config (Any): Configuration object.
            device (torch.device): target compute device (CPU or CUDA).
            validator (Optional[Any]): Standalone validator engine. Defaults to None.
            logger (Optional[Any]): DistillationLogger logger. Defaults to None.
            checkpoint_manager (Optional[Any]): CheckpointManager instance. Defaults to None.
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.loss_manager = loss_manager
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.validator = validator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

    def train(self) -> List[Dict[str, Any]]:
        """Execute the complete training pipeline across all epochs.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing averaged epoch loss metrics.
                Interface contract:
                [
                    {
                        "epoch": 1,
                        "pixel_loss": float,
                        "kd_loss": float,
                        "total_loss": float
                    },
                    ...
                ]
        """
        # Resolve epochs from config
        train_cfg = getattr(self.config, "training", None) or self.config
        epochs = int(getattr(train_cfg, "epochs", 300))

        logger.info(f"Starting distillation training for {epochs} epochs on device: {self.device}")

        # Check for existing checkpoint to resume training
        latest_checkpoint = None
        if self.checkpoint_manager is not None and getattr(self.checkpoint_manager, "checkpoint_dir", None) is not None:
            try:
                candidate = Path(self.checkpoint_manager.checkpoint_dir) / "latest.pth"
                if candidate.exists():
                    latest_checkpoint = candidate
            except Exception as e:
                logger.warning(f"Error checking checkpoint path: {e}")

        if latest_checkpoint is not None:
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            checkpoint_data = self.checkpoint_manager.load(latest_checkpoint)
            history: List[Dict[str, Any]] = checkpoint_data.get("training_history", [])
            restored_epoch = checkpoint_data.get("epoch", 0)
            start_epoch = restored_epoch + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.info("No checkpoint found. Starting training from epoch 1.")
            history = []
            start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            epoch_stats = self._train_epoch()

            # Execute scheduler step at the end of each epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # Run validation if validator is injected
            val_stats = {}
            if self.validator is not None:
                val_stats = self.validator.validate()

            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log statistics
            if self.logger is not None:
                self.logger.log_epoch(epoch, epoch_stats, val_stats, current_lr)
            else:
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Complete - "
                    f"Pixel Loss: {epoch_stats['pixel_loss']:.6f} | "
                    f"KD Loss: {epoch_stats['kd_loss']:.6f} | "
                    f"Total Loss: {epoch_stats['total_loss']:.6f}"
                )

            # Record epoch statistics
            stats_record = {
                "epoch": epoch,
                "pixel_loss": epoch_stats["pixel_loss"],
                "kd_loss": epoch_stats["kd_loss"],
                "total_loss": epoch_stats["total_loss"],
            }
            if "psnr" in val_stats:
                stats_record["psnr"] = val_stats["psnr"]
            if "ssim" in val_stats:
                stats_record["ssim"] = val_stats["ssim"]

            history.append(stats_record)

            # Save checkpoint if manager is injected
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(epoch, history, val_stats)

        return history

    def _train_epoch(self) -> Dict[str, float]:
        """Execute a single training epoch across all data loader batches.

        Returns:
            Dict[str, float]: Dict of epoch-averaged loss metrics.
        """
        # Explicitly enforce modes at the start of every epoch
        self.student_model.train()
        self.teacher_model.eval()

        running_pixel_loss = 0.0
        running_kd_loss = 0.0
        running_total_loss = 0.0
        num_batches = 0

        for lr_batch, hr_batch in self.train_loader:
            batch_stats = self._train_batch(lr_batch, hr_batch)

            running_pixel_loss += batch_stats["pixel_loss"]
            running_kd_loss += batch_stats["kd_loss"]
            running_total_loss += batch_stats["total_loss"]
            num_batches += 1

        if num_batches == 0:
            return {"pixel_loss": 0.0, "kd_loss": 0.0, "total_loss": 0.0}

        return {
            "pixel_loss": running_pixel_loss / num_batches,
            "kd_loss": running_kd_loss / num_batches,
            "total_loss": running_total_loss / num_batches,
        }

    def _train_batch(self, lr_batch: torch.Tensor, hr_batch: torch.Tensor) -> Dict[str, float]:
        """Execute training steps on a single batch of tensors.

        Moves tensors to device, zeroes out gradients, runs student/teacher forward,
        computes and backpropagates total loss, and steps the optimizer.

        Args:
            lr_batch (torch.Tensor): Low-resolution image batch tensor.
            hr_batch (torch.Tensor): Ground truth high-resolution image batch tensor.

        Returns:
            Dict[str, float]: Batch loss values.
        """
        # 1. Cast LR and HR tensors to device
        lr = lr_batch.to(self.device)  # lr: (B, 3, H, W)
        gt = hr_batch.to(self.device)  # gt: (B, 3, H*scale, W*scale)

        # 2. Reset gradients before forward executions (standard PyTorch sequence)
        self.optimizer.zero_grad()

        # 3. Student forward
        student_sr = self.student_model(lr)  # student_sr: (B, 3, H*scale, W*scale)

        # 4. Teacher forward
        # Teacher is evaluated under no_grad internally, but we reinforce it here
        with torch.no_grad():
            teacher_sr = self.teacher_model(lr)  # teacher_sr: (B, 3, H*scale, W*scale)

        # 5. Loss computation
        loss_dict = self.loss_manager(student_sr, teacher_sr, gt)
        total_loss = loss_dict["total_loss"]

        # 6. Backpropagation
        total_loss.backward()

        # 7. Optimizer step
        self.optimizer.step()

        return {
            "pixel_loss": loss_dict["pixel_loss"].item(),
            "kd_loss": loss_dict["kd_loss"].item(),
            "total_loss": total_loss.item(),
        }
