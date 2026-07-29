"""Loss package initializer exposing PixelLoss, KnowledgeDistillationLoss, and LossManager.

Author: Antigravity
Purpose: Expose modular losses.
"""

from training.losses.pixel_loss import PixelLoss
from training.losses.kd_loss import KnowledgeDistillationLoss
from training.losses.loss_manager import LossManager

__all__ = [
    "PixelLoss",
    "KnowledgeDistillationLoss",
    "LossManager",
]
