"""Pixel loss module for lightweight image super-resolution training.

Author: Antigravity
Purpose: Calculate raw pixel-level reconstruction loss.
"""

import torch
import torch.nn as nn


class PixelLoss(nn.Module):
    """Pixel reconstruction loss module.

    Computes raw, unweighted L1 loss between the student's super-resolved output
    and the ground truth high-resolution target.
    """

    def __init__(self) -> None:
        """Initialize PixelLoss."""
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, student_sr: torch.Tensor, gt_hr: torch.Tensor) -> torch.Tensor:
        """Compute pixel loss.

        Args:
            student_sr (torch.Tensor): Student super-resolved output tensor.
            gt_hr (torch.Tensor): Ground truth high-resolution image tensor.

        Returns:
            torch.Tensor: Scalar raw L1 loss.
        """
        # student_sr: (B, C, H, W), gt_hr: (B, C, H, W)
        loss = self.loss_fn(student_sr, gt_hr)
        # loss: scalar
        return loss
