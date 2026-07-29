"""Knowledge distillation loss module for image super-resolution distillation.

Author: Antigravity
Purpose: Calculate raw pixel-level distillation loss between student and teacher.
"""

import torch
import torch.nn as nn


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation pixel-level loss module.

    Computes raw, unweighted L1 loss between the student's super-resolved output
    and the teacher's super-resolved output.
    """

    def __init__(self) -> None:
        """Initialize KnowledgeDistillationLoss."""
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, student_sr: torch.Tensor, teacher_sr: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss.

        Args:
            student_sr (torch.Tensor): Student super-resolved output tensor.
            teacher_sr (torch.Tensor): Teacher super-resolved output tensor.

        Returns:
            torch.Tensor: Scalar raw L1 loss.
        """
        # student_sr: (B, C, H, W), teacher_sr: (B, C, H, W)
        loss = self.loss_fn(student_sr, teacher_sr)
        # loss: scalar
        return loss
