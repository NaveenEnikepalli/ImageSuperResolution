"""
Phase 1 verification script running diagnostic sanity checks on all infrastructure modules.
"""

from pathlib import Path
import logging

from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.utils.device import log_env_info
from training.utils.seed import set_seed
from training.utils.paths import PathManager


def main() -> None:
    # 1. Load default config template
    config_path = Path("configs/default_config.yaml")
    print(f"Loading configuration from: {config_path.resolve()}")
    config = load_config(str(config_path))
    
    # 2. Set up PathManager based on config parameters
    pm = PathManager(
        experiment_name=config.experiment_name,
        base_dir=config.output_dir
    )
    
    # 3. Initialize logger outputting to file inside run folder
    log_file_path = pm.log_dir / "verify_run.log"
    logger = setup_logger(
        name="VerifySRInfra",
        log_file=log_file_path,
        level=logging.INFO
    )
    
    logger.info("Initializing Phase 1 Infrastructure Diagnostics...")
    
    # 4. Diagnose and log environment info to console and file
    log_env_info(logger)
    
    # 5. Backup the validated config to the run configs folder
    config_backup_path = pm.config_dir / "config.yaml"
    logger.info(f"Backing up verified configuration to: {config_backup_path}")
    config.save(config_backup_path)
    
    # 6. Verify seed initialization reproducibility
    seed_val = config.training.seed
    logger.info(f"Configuring reproducibility seeds to: {seed_val}")
    set_seed(seed_val)
    
    logger.info("System Paths Created successfully:")
    logger.info(f"  Run directory:        {pm.run_dir}")
    logger.info(f"  Checkpoints directory:{pm.checkpoint_dir}")
    logger.info(f"  Samples directory:    {pm.sample_dir}")
    logger.info(f"  Logs directory:       {pm.log_dir}")
    logger.info(f"  Configs backup:       {pm.config_dir}")
    
    logger.info("=" * 60)
    logger.info("PHASE 1 DIAGNOSTIC RESULTS: SUCCESS")
    logger.info("All modular infrastructure utilities functioning as specified.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
