"""Loss manager module orchestrating loss calculations and configuration-driven weights.

Author: Antigravity
Purpose: Combine multiple weighted losses into a single total loss dictionary.
"""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn

from training.losses.pixel_loss import PixelLoss
from training.losses.kd_loss import KnowledgeDistillationLoss


class LossManager(nn.Module):
    """Loss manager class responsible for aggregating weighted loss functions.

    Maintains standard PixelLoss and KnowledgeDistillationLoss modules, applies
    configurable weights retrieved from configuration objects or dictionaries, and
    computes the total loss. Designed to be extensible for future losses.
    """

    def __init__(self, loss_config: Optional[Union[Dict[str, Any], Any]] = None) -> None:
        """Initialize LossManager.

        Args:
            loss_config (Optional[Union[Dict[str, Any], Any]]): Configuration dictionary
                or object containing weight overrides. Defaults to None.
        """
        super().__init__()

        # Default configuration values
        self.pixel_weight = 1.0
        self.kd_weight = 0.2

        # Parse config weights if provided
        if loss_config is not None:
            self._parse_config(loss_config)

        # Composition using nn.ModuleDict for standard sub-module tracking and casting
        self.loss_functions = nn.ModuleDict({
            "pixel_loss": PixelLoss(),
            "kd_loss": KnowledgeDistillationLoss()
        })

    def _parse_config(self, loss_config: Union[Dict[str, Any], Any]) -> None:
        """Extract loss weights from configuration inputs recursively.

        Args:
            loss_config (Union[Dict[str, Any], Any]): Config dict or object.
        """
        if isinstance(loss_config, dict):
            # 1. Direct dictionary parsing
            self.pixel_weight = float(loss_config.get("pixel_weight", self.pixel_weight))
            self.kd_weight = float(loss_config.get("kd_weight", self.kd_weight))
        else:
            # 2. ExperimentConfig object structure parsing
            # Handles experiment-wide losses configurations:
            #   losses:
            #     pixel_loss: "l1"
            #     distillation:
            #       alpha_pixel: 1.0
            #       beta_feature: 0.1
            losses_attr = getattr(loss_config, "losses", None)
            distillation_attr = getattr(loss_config, "distillation", None)

            if losses_attr is not None:
                if isinstance(losses_attr, dict):
                    distill_dict = losses_attr.get("distillation", {})
                    if isinstance(distill_dict, dict):
                        self.pixel_weight = float(distill_dict.get("alpha_pixel", self.pixel_weight))
                        self.kd_weight = float(distill_dict.get("beta_feature", self.kd_weight))
                else:
                    distill_cfg = getattr(losses_attr, "distillation", None)
                    if distill_cfg is not None:
                        self.pixel_weight = float(getattr(distill_cfg, "alpha_pixel", self.pixel_weight))
                        self.kd_weight = float(getattr(distill_cfg, "beta_feature", self.kd_weight))
            elif distillation_attr is not None:
                # Handles default_config.yaml structure (distillation as top-level object)
                self.pixel_weight = float(getattr(distillation_attr, "alpha_pixel", self.pixel_weight))
                self.kd_weight = float(getattr(distillation_attr, "beta_feature", self.kd_weight))

    def forward(
        self,
        student_sr: torch.Tensor,
        teacher_sr: torch.Tensor,
        gt_hr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all registered losses, apply weights, and sum them.

        Args:
            student_sr (torch.Tensor): Student super-resolved tensor.
            teacher_sr (torch.Tensor): Teacher super-resolved tensor.
            gt_hr (torch.Tensor): Ground truth high-resolution image.

        Returns:
            Dict[str, torch.Tensor]: Stable interface dictionary containing:
                - "pixel_loss": raw, unweighted PixelLoss tensor.
                - "kd_loss": raw, unweighted KnowledgeDistillationLoss tensor.
                - "total_loss": weighted total loss sum.
        """
        # Calculate raw unweighted losses
        pixel_loss = self.loss_functions["pixel_loss"](student_sr, gt_hr)
        kd_loss = self.loss_functions["kd_loss"](student_sr, teacher_sr)

        # Apply weights to compute total loss
        total_loss = (self.pixel_weight * pixel_loss) + (self.kd_weight * kd_loss)

        return {
            "pixel_loss": pixel_loss,
            "kd_loss": kd_loss,
            "total_loss": total_loss
        }

    def extra_repr(self) -> str:
        """Provide parameters descriptor details.

        Returns:
            str: Introspection parameters descriptor.
        """
        return f"pixel_weight={self.pixel_weight}, kd_weight={self.kd_weight}"
