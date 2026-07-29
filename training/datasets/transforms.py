"""
Modular image transforms and augmentations for Super-Resolution.
Operates on PIL Images to maintain computational speed and flexibility.
"""

import random
from typing import List, Callable, Tuple
from PIL import Image, ImageEnhance


class Compose:
    """Compose multiple image transforms sequentially."""

    def __init__(self, transforms: List[Callable[[Image.Image], Image.Image]]) -> None:
        """Initialize Compose with a list of callable transforms.

        Args:
            transforms: List of transforms to apply sequentially.
        """
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply all transforms sequentially to the input image.

        Args:
            img: Input PIL Image.

        Returns:
            Image.Image: Transformed PIL Image.
        """
        for t in self.transforms:
            img = t(img)
        return img


class RandomCrop:
    """Extract a random square or rectangular patch of size `patch_size` from an image."""

    def __init__(self, size: int) -> None:
        """Initialize RandomCrop with target patch size.

        Args:
            size: Width and height of the extracted patch.
        """
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        """Crop a random patch from the image.

        Args:
            img: High-resolution PIL Image.

        Returns:
            Image.Image: Cropped patch.
        """
        w, h = img.size
        if w < self.size or h < self.size:
            # Upscale image using bicubic interpolation if it is smaller than patch size
            new_w = max(w, self.size)
            new_h = max(h, self.size)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - self.size)
        top = random.randint(0, h - self.size)
        return img.crop((left, top, left + self.size, top + self.size))


class CenterCrop:
    """Extract a center crop of size `patch_size` from an image."""

    def __init__(self, size: int) -> None:
        """Initialize CenterCrop with target patch size.

        Args:
            size: Width and height of the extracted center patch.
        """
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        """Crop the center of the image.

        Args:
            img: PIL Image.

        Returns:
            Image.Image: Cropped center patch.
        """
        w, h = img.size
        if w < self.size or h < self.size:
            new_w = max(w, self.size)
            new_h = max(h, self.size)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size

        left = (w - self.size) // 2
        top = (h - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    """Flip image horizontally with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize RandomHorizontalFlip.

        Args:
            p: Probability of applying flip. Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply horizontal flip dynamically.

        Args:
            img: PIL Image.

        Returns:
            Image.Image: Flipped or original image.
        """
        if random.random() < self.p:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img


class RandomVerticalFlip:
    """Flip image vertically with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize RandomVerticalFlip.

        Args:
            p: Probability of applying flip. Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply vertical flip dynamically.

        Args:
            img: PIL Image.

        Returns:
            Image.Image: Flipped or original image.
        """
        if random.random() < self.p:
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return img


class RandomRotation90:
    """Rotate image by 90, 180, or 270 degrees randomly to preserve pixel grid structure."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize RandomRotation90.

        Args:
            p: Probability of applying rotation. Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random 90-degree based rotation.

        Args:
            img: PIL Image.

        Returns:
            Image.Image: Rotated or original image.
        """
        if random.random() < self.p:
            angle = random.choice([
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270
            ])
            return img.transpose(angle)
        return img


class ColorJitter:
    """Apply random adjustments to brightness, contrast, and saturation of the image."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        p: float = 0.2
    ) -> None:
        """Initialize ColorJitter.

        Args:
            brightness: Max brightness adjustment factor. Defaults to 0.1.
            contrast: Max contrast adjustment factor. Defaults to 0.1.
            saturation: Max saturation adjustment factor. Defaults to 0.1.
            p: Probability of applying color jitter. Defaults to 0.2.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply color enhancements dynamically.

        Args:
            img: PIL Image.

        Returns:
            Image.Image: Color-adjusted image.
        """
        if random.random() < self.p:
            if self.brightness > 0:
                factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
                img = ImageEnhance.Brightness(img).enhance(factor)
            if self.contrast > 0:
                factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
                img = ImageEnhance.Contrast(img).enhance(factor)
            if self.saturation > 0:
                factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
                img = ImageEnhance.Color(img).enhance(factor)
        return img


def get_transforms(patch_size: int, is_train: bool = True) -> Compose:
    """Factory helper to build transformation pipelines.

    Args:
        patch_size: Square dimension of the target crop patch.
        is_train: If True, returns training augmentations. Defaults to True.

    Returns:
        Compose: Instantiated transformation compose pipeline.
    """
    if is_train:
        return Compose([
            RandomCrop(patch_size),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation90(p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.2)
        ])
    else:
        return Compose([
            CenterCrop(patch_size)
        ])
