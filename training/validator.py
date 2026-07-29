"""Validation engine for evaluating Student Model super-resolution performance.

Author: Antigravity
Purpose: Coordinates model evaluation using metrics with no training or teacher dependencies.
"""

import logging
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics.average_meter import AverageMeter

logger = logging.getLogger(__name__)


class Validator:
    """Validation engine orchestrator for evaluating Super-Resolution quality.

    Runs validation batches using the student model under eval and no_grad contexts,
    evaluates outputs using injected PSNR and SSIM metrics wrapper instances, aggregates
    scores via AverageMeters, and returns averaged epoch statistics.

    Example:
        validator = Validator(
            student_model=student,
            validation_loader=val_loader,
            device=device,
            psnr_metric=PSNRMetric(),
            ssim_metric=SSIMMetric()
        )
        val_results = validator.validate()
        print(val_results["psnr"])
    """

    def __init__(
        self,
        student_model: nn.Module,
        validation_loader: DataLoader,
        device: torch.device,
        psnr_metric: Any,
        ssim_metric: Any,
    ) -> None:
        """Initialize the Validator.

        Args:
            student_model (nn.Module): Student model being evaluated.
            validation_loader (DataLoader): Validation loader yielding LR/HR pairs.
            device (torch.device): Compute device (CPU or CUDA).
            psnr_metric (Any): Injected PSNR evaluation metric wrapper instance.
            ssim_metric (Any): Injected SSIM evaluation metric wrapper instance.
        """
        self.student_model = student_model
        self.validation_loader = validation_loader
        self.device = device
        self.psnr_metric = psnr_metric
        self.ssim_metric = ssim_metric

    def validate(self) -> Dict[str, float]:
        """Perform evaluation over the entire validation dataset.

        Returns:
            Dict[str, float]: Stable API dictionary containing aggregated stats.
                Interface contract:
                {
                    "psnr": float,
                    "ssim": float
                }
        """
        logger.info("Executing validation evaluation loop...")

        # Explicitly enforce eval mode
        self.student_model.eval()

        # Instantiate metrics meters
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        # Execute evaluation inside no_grad context to disable gradient tracking
        with torch.no_grad():
            for lr_batch, hr_batch in self.validation_loader:
                self._validate_batch(lr_batch, hr_batch, psnr_meter, ssim_meter)

        logger.info(
            f"Validation Completed - "
            f"Average PSNR: {psnr_meter.average:.4f} dB | "
            f"Average SSIM: {ssim_meter.average:.4f}"
        )

        return {
            "psnr": psnr_meter.average,
            "ssim": ssim_meter.average,
        }

    def _validate_batch(
        self,
        lr_batch: torch.Tensor,
        hr_batch: torch.Tensor,
        psnr_meter: AverageMeter,
        ssim_meter: AverageMeter,
    ) -> None:
        """Execute evaluation on a single validation batch.

        Moves inputs, runs model inference, calls metrics wrapper functions, and updates
        their respective average meters.

        Args:
            lr_batch (torch.Tensor): Low-resolution image batch.
            hr_batch (torch.Tensor): Ground truth high-resolution image batch.
            psnr_meter (AverageMeter): Metric accumulator for PSNR values.
            ssim_meter (AverageMeter): Metric accumulator for SSIM values.
        """
        # Move LR and HR tensors to device
        lr = lr_batch.to(self.device)  # lr: (B, 3, H, W)
        gt = hr_batch.to(self.device)  # gt: (B, 3, H*scale, W*scale)
        batch_size = lr.shape[0]

        # Student model inference
        pred = self.student_model(lr)  # pred: (B, 3, H*scale, W*scale)

        # Metric evaluations
        psnr_val = self.psnr_metric(pred, gt)
        ssim_val = self.ssim_metric(pred, gt)

        # Update average meters
        psnr_meter.update(psnr_val, n=batch_size)
        ssim_meter.update(ssim_val, n=batch_size)
