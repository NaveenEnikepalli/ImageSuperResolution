"""Orchestrator script for starting distillation training on Student Model.

Author: Antigravity
Purpose: Parse configuration, build dataset and loaders, instantiate student, teacher,
and loss manager, create optimizer and scheduler, construct Trainer, and run.
"""

import sys
import os
import logging
from pathlib import Path
import torch

from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.utils.seed import set_random_seed
from training.datasets.dataloader import create_train_loader
from training.models.student_model import StudentModel
from training.teacher.swinir_wrapper import SwinIRWrapper
from training.losses.loss_manager import LossManager
from training.trainer import Trainer

logger = logging.getLogger("TrainEntry")


def main() -> None:
    """Load components, perform dependency injection, and execute training loop."""
    # 1. Setup logging context
    setup_logger(name="training", level=logging.INFO)

    # 2. Parse configuration file path
    config_path = "training/configs/default_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration file from: {config_path}")
    config = load_config(config_path)

    # 3. Set reproducibility seed
    # Check both config.training.seed and config.random_seed
    train_cfg = getattr(config, "training", None) or config
    seed = getattr(train_cfg, "seed", getattr(config, "random_seed", 42))
    set_random_seed(seed)
    logger.info(f"Reproducibility seed set to: {seed}")

    # 4. Determine compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution compute device: {device}")

    # 5. Build Training DataLoader
    logger.info("Initializing dataset training data loader...")
    train_loader = create_train_loader(config)

    # 6. Instantiate Models
    m_cfg = config.model
    logger.info("Instantiating StudentModel architecture...")
    student = StudentModel(
        in_channels=getattr(m_cfg, "in_channels", 3),
        out_channels=3,
        num_features=getattr(m_cfg, "num_channels", 48),
        distilled_channels=getattr(m_cfg, "distilled_channels", 24),
        num_blocks=getattr(m_cfg, "num_blocks", 3),
        scale=getattr(m_cfg, "scale", 4),
        negative_slope=getattr(m_cfg, "negative_slope", 0.05),
    )
    student.to(device)

    logger.info("Instantiating SwinIRWrapper teacher model...")
    teacher = SwinIRWrapper(
        scale=getattr(m_cfg, "scale", 4),
        in_channels=getattr(m_cfg, "in_channels", 3),
    )
    teacher.to(device)

    # Load teacher checkpoint weights if configured
    checkpoint_path = getattr(m_cfg, "checkpoint_path", None)
    if checkpoint_path:
        teacher.load_checkpoint(checkpoint_path)

    # 7. Instantiate Loss Framework
    logger.info("Instantiating LossManager configuration...")
    loss_manager = LossManager(loss_config=config)
    loss_manager.to(device)

    # 8. Create Optimizer (Adam, configuration-driven)
    lr = float(getattr(train_cfg, "lr", getattr(config, "learning_rate", 2e-4)))
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    opt_cfg = getattr(train_cfg, "optimizer", None)
    if isinstance(opt_cfg, dict):
        lr = float(opt_cfg.get("lr", lr))
        weight_decay = float(opt_cfg.get("weight_decay", weight_decay))
    elif opt_cfg is not None and not isinstance(opt_cfg, str):
        lr = float(getattr(opt_cfg, "lr", lr))
        weight_decay = float(getattr(opt_cfg, "weight_decay", weight_decay))

    logger.info(f"Creating optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
    optimizer = torch.optim.Adam(
        params=student.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 9. Create Scheduler (CosineAnnealingLR, configuration-driven)
    epochs = int(getattr(train_cfg, "epochs", 300))
    sched_cfg = getattr(train_cfg, "scheduler", None)
    t_max = epochs
    eta_min = 1e-6

    if isinstance(sched_cfg, dict):
        t_max = int(sched_cfg.get("T_max", t_max))
        eta_min = float(sched_cfg.get("eta_min", eta_min))
    elif sched_cfg is not None:
        t_max = int(getattr(sched_cfg, "T_max", t_max))
        eta_min = float(getattr(sched_cfg, "eta_min", eta_min))

    logger.info(f"Creating scheduler: CosineAnnealingLR (T_max={t_max}, eta_min={eta_min})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=t_max,
        eta_min=eta_min
    )

    # 10. Construct Trainer via Dependency Injection
    logger.info("Constructing Trainer via dependency injection...")
    trainer = Trainer(
        student_model=student,
        teacher_model=teacher,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        config=config,
        device=device
    )

    # 11. Run Distillation Training Loops
    logger.info("Starting training engine execution...")
    _ = trainer.train()
    logger.info("Distillation training completed successfully.")


if __name__ == "__main__":
    main()
