"""PSNR metric wrapper around PyTorch Image Quality (PIQ).

Author: Antigravity
Purpose: Wrap PIQ PSNR calculation. Tensors are clamped to range [0.0, 1.0]
to prevent numerical drift and ensure compatibility.
"""

import torch
import piq


class PSNRMetric:
    """Peak Signal-to-Noise Ratio (PSNR) metric calculator wrapper.

    Acts as an isolated facade over PIQ library functions to prevent leaking
    implementation details.
    """

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR between prediction and target tensors.

        Before evaluation, tensors are clamped to [0.0, 1.0] range to prevent
        numerical drift.

        Args:
            prediction (torch.Tensor): Super-resolved image output tensor (B, C, H, W).
            target (torch.Tensor): Ground truth high-resolution target tensor (B, C, H, W).

        Returns:
            float: PSNR value.
        """
        # Clamp tensors to configured range [0.0, 1.0] before calling PIQ
        pred = torch.clamp(prediction, min=0.0, max=1.0)
        tgt = torch.clamp(target, min=0.0, max=1.0)

        # Call PIQ library function
        score = piq.psnr(pred, tgt, data_range=1.0)
        return float(score.item())
