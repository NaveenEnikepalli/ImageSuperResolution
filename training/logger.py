"""Logging utility for tracking training and validation progress.

Author: Antigravity
Purpose: Formats and logs statistics to the console and an optional text file.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class DistillationLogger:
    """Logger tracking console printouts and file-based training updates.

    Exposes a stable API log_epoch() and supports writing records to a
    local log file log.txt.

    Example:
        logger_instance = DistillationLogger(log_dir="outputs/runs")
        logger_instance.log_epoch(
            epoch=1,
            train_stats={"pixel_loss": 0.02, "kd_loss": 0.01, "total_loss": 0.022},
            val_stats={"psnr": 29.5, "ssim": 0.85},
            lr=1e-4
        )
    """

    def __init__(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize DistillationLogger.

        Args:
            log_dir (Optional[Union[str, Path]]): Directory to store log.txt.
                If None, logs to console only.
        """
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "log.txt"
        else:
            self.log_file = None

    def log_epoch(
        self,
        epoch: int,
        train_stats: Dict[str, float],
        val_stats: Dict[str, float],
        lr: float,
    ) -> None:
        """Log epoch statistics to the console and optional log file.

        Args:
            epoch (int): The current training epoch.
            train_stats (Dict[str, float]): Training statistics (losses).
            val_stats (Dict[str, float]): Validation statistics (metrics).
            lr (float): The current learning rate.
        """
        # Format the epoch logging output strings
        header = f"Epoch: {epoch} | LR: {lr:.6f}"
        
        train_pixel = train_stats.get("pixel_loss", 0.0)
        train_kd = train_stats.get("kd_loss", 0.0)
        train_total = train_stats.get("total_loss", 0.0)
        train_str = f"  Train Pixel Loss: {train_pixel:.6f} | KD Loss: {train_kd:.6f} | Total Loss: {train_total:.6f}"
        
        val_psnr = val_stats.get("psnr", 0.0)
        val_ssim = val_stats.get("ssim", 0.0)
        val_str = f"  Val PSNR: {val_psnr:.4f} dB | SSIM: {val_ssim:.4f}"

        # Combine into log record block
        log_block = f"{header}\n{train_str}\n{val_str}\n"

        # Log to standard output stream (using Python logger)
        logger.info(header)
        logger.info(train_str)
        logger.info(val_str)

        # Log to text file if enabled
        if self.log_file is not None:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_block)
            except Exception as e:
                logger.error(f"Failed to write to log file {self.log_file}. Error: {e}")
