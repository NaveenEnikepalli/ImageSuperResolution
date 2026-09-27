"""Isolated training orchestration script for Student x2 Super-Resolution model.

Author: Antigravity
Purpose: Parse Student x2 training configuration, instantiate StudentModel with scale=2,
set up dataset, loss manager, trainer, checkpoint manager, and run training pipeline.
Outputs checkpoints to checkpoints/student/student_x2.pth.
"""

import sys
import os
import shutil
import logging
from pathlib import Path
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.utils.seed import set_random_seed
from training.utils.paths import PathManager
from training.utils.checkpoint import CheckpointManager
from training.logger import DistillationLogger
from training.datasets.dataloader import create_train_loader, create_validation_loader
from training.models.student_model import StudentModel
from training.teacher.swinir_wrapper import SwinIRWrapper
from training.losses.loss_manager import LossManager
from training.metrics.psnr import PSNRMetric
from training.metrics.ssim import SSIMMetric
from training.validator import Validator
from training.trainer import Trainer

logger = logging.getLogger("TrainStudentX2")


def main() -> None:
    """Load components, perform dependency injection, and execute Student x2 training loop."""
    # 1. Parse configuration file path
    config_path = "training/configs/student_x2_training.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Force scale=2 in configuration
    if hasattr(config, "model") and hasattr(config.model, "scale"):
        config.model.scale = 2

    # 2. Setup output paths via PathManager
    experiment_name = getattr(config, "experiment_name", "student_x2")
    output_dir = getattr(config, "output_dir", "outputs")
    path_mgr = PathManager(experiment_name=experiment_name, base_dir=output_dir)

    # 3. Setup logging context
    log_file = path_mgr.log_dir / "train_x2.log"
    setup_logger(name="training", log_file=log_file, level=logging.INFO)
    logger.info(f"Loading Student x2 configuration from: {config_path}")
    logger.info(f"Experiment workspace output directory: {path_mgr.run_dir}")

    # 4. Set reproducibility seed
    train_cfg = getattr(config, "training", None) or config
    seed = getattr(train_cfg, "seed", getattr(config, "random_seed", 42))
    set_random_seed(seed)
    logger.info(f"Reproducibility seed set to: {seed}")

    # 5. Determine compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution compute device: {device}")

    # 6. Build Training & Validation DataLoaders for scale=2
    logger.info("Initializing dataset training data loader for scale=2...")
    train_loader = create_train_loader(config)

    logger.info("Initializing validation dataset data loaders for scale=2...")
    val_loaders = create_validation_loader(config)
    val_loader = None
    if val_loaders:
        val_loader = val_loaders.get("Set5") or next(iter(val_loaders.values()))

    # 7. Instantiate StudentModel with scale=2
    m_cfg = getattr(config, "student", None) or getattr(config, "model", None)
    logger.info("Instantiating StudentModel architecture with scale=2...")
    student = StudentModel(
        in_channels=getattr(m_cfg, "in_channels", 3),
        out_channels=getattr(m_cfg, "out_channels", 3),
        num_features=getattr(m_cfg, "num_channels", 48),
        distilled_channels=getattr(m_cfg, "distilled_channels", 24),
        num_blocks=getattr(m_cfg, "num_blocks", 3),
        scale=2,  # Explicitly scale=2
        negative_slope=getattr(m_cfg, "negative_slope", 0.05),
    )
    student.to(device)

    # Teacher model setup for x2 (using SwinIR scale=2 if enabled)
    t_cfg = getattr(config, "teacher", None)
    teacher_enabled = getattr(t_cfg, "enable", False) if t_cfg else False
    teacher = SwinIRWrapper(scale=2, in_channels=3).to(device)

    if teacher_enabled and t_cfg is not None:
        ckpt_path = getattr(t_cfg, "checkpoint_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Loading pretrained teacher scale=2 checkpoint: {ckpt_path}")
            teacher.load_checkpoint(ckpt_path)

    # 8. Instantiate Loss Framework
    logger.info("Instantiating LossManager configuration...")
    loss_manager = LossManager(loss_config=config)
    loss_manager.to(device)

    # 9. Create Optimizer
    lr = float(getattr(train_cfg, "lr", getattr(config, "learning_rate", 1e-4)))
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    logger.info(f"Creating optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
    optimizer = torch.optim.Adam(
        params=student.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 10. Create Scheduler
    epochs = int(getattr(train_cfg, "epochs", getattr(config, "epochs", 100)))
    milestones = getattr(train_cfg, "lr_milestones", [50, 80])
    gamma = float(getattr(train_cfg, "lr_gamma", 0.5))
    logger.info(f"Creating scheduler: MultiStepLR (milestones={milestones}, gamma={gamma})")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(milestones),
        gamma=gamma
    )

    # 11. Create DistillationLogger, CheckpointManager, and Validator
    dist_logger = DistillationLogger(log_dir=path_mgr.log_dir)

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=path_mgr.checkpoint_dir,
        student_model=student,
        optimizer=optimizer,
        scheduler=scheduler
    )

    validator = None
    if val_loader is not None:
        validator = Validator(
            student_model=student,
            validation_loader=val_loader,
            device=device,
            psnr_metric=PSNRMetric(),
            ssim_metric=SSIMMetric()
        )
        logger.info("Validator constructed and injected into Student x2 Trainer pipeline.")

    # 12. Construct Trainer via Dependency Injection
    logger.info("Constructing Trainer for Student x2...")
    trainer = Trainer(
        student_model=student,
        teacher_model=teacher,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        config=config,
        device=device,
        validator=validator,
        logger=dist_logger,
        checkpoint_manager=checkpoint_mgr
    )

    # 13. Execute Training
    logger.info("Starting Student x2 training execution...")
    history = trainer.train()
    logger.info("Student x2 training completed successfully.")

    # 14. Export final/best checkpoint to checkpoints/student/student_x2.pth
    target_ckpt_dir = Path("checkpoints/student")
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt_path = target_ckpt_dir / "student_x2.pth"

    best_source = path_mgr.checkpoint_dir / "best.pth"
    latest_source = path_mgr.checkpoint_dir / "latest.pth"

    source_path = best_source if best_source.exists() else latest_source
    if source_path.exists():
        shutil.copy(source_path, target_ckpt_path)
        logger.info(f"Exported trained Student x2 model checkpoint to: {target_ckpt_path}")
    else:
        # Save current state directly if output files not present
        torch.save(student.state_dict(), target_ckpt_path)
        logger.info(f"Saved Student x2 state dict directly to: {target_ckpt_path}")


if __name__ == "__main__":
    main()
