"""
Custom centralized logger configuration supporting synchronized console and file output.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ImageSR",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure a unified logger with synchronized console and file handlers.

    Ensures single registration of handlers to avoid duplicate output.

    Args:
        name: Name of the logger workspace namespace.
        log_file: Path to write logging outputs to. If None, only console is used.
        level: Logging level filter threshold. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent logs from propagating to root logger to avoid duplicate prints in notebooks
    logger.propagate = False

    # Clear pre-existing handlers if initialized previously
    if logger.handlers:
        logger.handlers.clear()

    # Formatter setups
    console_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    file_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 2. File Handler (if file destination specified)
    if log_file is not None:
        try:
            # Ensure folder structure exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Unable to write logs to file destination: {log_file}. Error: {e}")

    return logger
