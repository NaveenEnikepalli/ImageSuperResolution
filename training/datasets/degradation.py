"""
Degradation pipeline for generating Low-Resolution (LR) images dynamically from High-Resolution (HR) inputs.
"""

from io import BytesIO
from typing import Tuple
from PIL import Image
import torch
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import torchvision.transforms.functional as TF


class DegradationPipeline:
    """Applies Gaussian Blur, Bicubic Downsampling, JPEG Compression, and Noise dynamically."""

    def __init__(
        self,
        scale: int,
        blur_kernel: int = 21,
        blur_sigma: float = 0.0,
        noise_std: float = 0.0,
        jpeg_quality: int = 100
    ) -> None:
        """Initialize the degradation pipeline.

        Args:
            scale: Downscaling upsample factor (choices: 2, 4).
            blur_kernel: Width of Gaussian kernel. Must be positive and odd.
            blur_sigma: Standard deviation for Gaussian kernel. If 0.0, blur is skipped.
            noise_std: Standard deviation of additive Gaussian noise. If 0.0, skipped.
            jpeg_quality: JPEG compression factor (1-100). If 100, compression is skipped.
        """
        self.scale = scale
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality

        # Validate parameters on load
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be positive and odd, got {self.blur_kernel}")
        if self.blur_sigma < 0.0:
            raise ValueError(f"blur_sigma cannot be negative, got {self.blur_sigma}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std cannot be negative, got {self.noise_std}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")

    def _apply_jpeg_compression(self, tensor: torch.Tensor) -> torch.Tensor:
        """Helper to simulate JPEG compression using BytesIO buffer.

        Args:
            tensor: PyTorch float tensor in range [0.0, 1.0].

        Returns:
            torch.Tensor: Compressed float tensor in range [0.0, 1.0].
        """
        # Convert tensor back to PIL Image
        # 1. Clamp to safe bounds and scale to uint8 range
        clamped = torch.clamp(tensor, 0.0, 1.0)
        img_np = (clamped.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        pil_img = Image.fromarray(img_np)

        # 2. Save to virtual file with compressed quality
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)
        
        # 3. Reload image and convert back to normalized PyTorch tensor
        compressed_img = Image.open(buffer)
        return TF.to_tensor(compressed_img)

    def __call__(self, hr_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert HR image to normalized tensor and dynamically generate corresponding LR tensor.

        Args:
            hr_img: High-resolution PIL Image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated
                (LR_tensor, HR_tensor), both normalized to range [0.0, 1.0].
        """
        # Convert HR PIL Image to PyTorch Tensor immediately
        hr_tensor = TF.to_tensor(hr_img)
        
        # Determine dynamic target LR dimensions
        hr_h, hr_w = hr_tensor.shape[1], hr_tensor.shape[2]
        lr_h = hr_h // self.scale
        lr_w = hr_w // self.scale

        # Make copy of HR tensor to construct LR tensor
        lr_tensor = hr_tensor.clone()

        # Step 1: Gaussian Blur
        if self.blur_sigma > 0.0:
            # TF.gaussian_blur expects tensor shape (..., C, H, W)
            lr_tensor = TF.gaussian_blur(
                lr_tensor,
                kernel_size=[self.blur_kernel, self.blur_kernel],
                sigma=[self.blur_sigma, self.blur_sigma]
            )

        # Step 2: Downsample via Bicubic Interpolation
        # F.interpolate requires (B, C, H, W) batch dimension
        lr_tensor = F.interpolate(
            lr_tensor.unsqueeze(0),
            size=(lr_h, lr_w),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)

        # Step 3: Optional JPEG Compression
        if self.jpeg_quality < 100:
            lr_tensor = self._apply_jpeg_compression(lr_tensor)

        # Step 4: Optional Additive Gaussian Noise
        if self.noise_std > 0.0:
            noise = torch.randn_like(lr_tensor) * self.noise_std
            lr_tensor = torch.clamp(lr_tensor + noise, 0.0, 1.0)

        return lr_tensor, hr_tensor
