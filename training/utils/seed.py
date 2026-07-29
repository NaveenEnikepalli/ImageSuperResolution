"""
Reproducibility and seeding utilities to guarantee deterministic training execution.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for Python, NumPy, and PyTorch to guarantee reproducibility.

    Args:
        seed: The integer seed value to initialize generators with.
        deterministic: If True, configures CuDNN backends to be deterministic
            (benchmark=False, deterministic=True). This might affect training speed.
    """
    # 1. Standard python random seed
    random.seed(seed)

    # 2. NumPy random seed
    np.random.seed(seed)

    # 3. PyTorch random seeds (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. Deterministic configuration for CUDA backends
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allows CuDNN to dynamically find optimized algorithms (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_random_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch CPU/CUDA backends.

    Args:
        seed (int): Seeding value to guarantee reproducibility.
    """
    set_seed(seed, deterministic=True)

