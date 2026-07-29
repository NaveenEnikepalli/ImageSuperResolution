"""Package initializer exposing core modules of the Student model.

Author: Antigravity
Purpose: Expose student modular architectures.
"""

from training.models.modules.feature_extractor import FeatureExtractor
from training.models.modules.rfdb import ResidualFeatureDistillationBlock
from training.models.modules.global_feature_fusion import GlobalFeatureFusion
from training.models.modules.pixelshuffle import PixelShuffle
from training.models.modules.reconstruction import Reconstruction

__all__ = [
    "FeatureExtractor",
    "ResidualFeatureDistillationBlock",
    "GlobalFeatureFusion",
    "PixelShuffle",
    "Reconstruction",
]
