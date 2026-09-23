"""
Unit tests validating Phase 1 project infrastructure.
"""

import logging
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from training.utils.constants import (
    SUPPORTED_DATASETS,
    SUPPORTED_SCALES,
    DEFAULT_EXPERIMENT_NAME
)
from training.utils.seed import set_seed
from training.utils.device import get_device, log_env_info
from training.utils.paths import PathManager
from training.utils.checkpoint import CheckpointManager
from training.utils.config import load_config, ExperimentConfig


def test_constants() -> None:
    """Verify that vital constants are defined correctly."""
    assert "DIV2K" in SUPPORTED_DATASETS
    assert 2 in SUPPORTED_SCALES
    assert 4 in SUPPORTED_SCALES
    assert DEFAULT_EXPERIMENT_NAME == "default_distill_run"


def test_reproducibility() -> None:
    """Verify that set_seed ensures reproducibility across random generators."""
    seed_value = 100
    
    # Run 1
    set_seed(seed_value, deterministic=True)
    arr1 = np.random.rand(10)
    ten1 = torch.rand(10)

    # Run 2
    set_seed(seed_value, deterministic=True)
    arr2 = np.random.rand(10)
    ten2 = torch.rand(10)

    assert np.allclose(arr1, arr2)
    assert torch.allclose(ten1, ten2)


def test_device_detection() -> None:
    """Verify compute device acquisition and debug logging execution."""
    device = get_device()
    assert isinstance(device, torch.device)
    
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    # Ensure it executes without throwing exceptions
    log_env_info(logger)


def test_path_manager() -> None:
    """Verify path manager directories creation and path properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_name = "test_run_dirs"
        pm = PathManager(experiment_name=exp_name, base_dir=tmpdir)

        base_path = Path(tmpdir)
        run_dir = base_path / "runs" / exp_name
        
        assert pm.run_dir == run_dir
        assert pm.checkpoint_dir == run_dir / "checkpoints"
        assert pm.log_dir == run_dir / "logs"
        assert pm.sample_dir == run_dir / "samples"
        assert pm.config_dir == run_dir / "configs"

        # Assert folders were actually created on disk
        assert pm.checkpoint_dir.is_dir()
        assert pm.log_dir.is_dir()
        assert pm.sample_dir.is_dir()
        assert pm.config_dir.is_dir()


def test_config_parsing_and_saving() -> None:
    """Verify load_config loads, validates, and serializes correctly."""
    config_dict = {
        "experiment_name": "test_exp",
        "model": {
            "type": "student",
            "name": "edsr_small",
            "scale": 2,
            "num_blocks": 4,
            "num_channels": 32
        },
        "dataset": {
            "name": "Set5",
            "patch_size": 96,
            "batch_size": 8
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / "test_config.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

        # 1. Load config
        config = load_config(str(yaml_file))
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "test_exp"
        assert config.model.type == "student"
        assert config.model.scale == 2
        assert config.model.num_blocks == 4
        assert config.dataset.name == "Set5"
        assert config.dataset.patch_size == 96
        assert config.dataset.batch_size == 8

        # Validate defaults were retained
        assert config.model.num_channels == 32
        assert config.training.epochs == 100  # Default value
        
        # 2. Save config backup
        backup_file = Path(tmpdir) / "config_backup.yaml"
        config.save(backup_file)
        assert backup_file.is_file()

        # Load backup and verify identity
        backup_config = load_config(str(backup_file))
        assert backup_config.experiment_name == config.experiment_name
        assert backup_config.model.scale == config.model.scale


def test_config_validation_failures() -> None:
    """Verify invalid configuration properties raise appropriate exceptions."""
    # Invalid model type
    c1 = ExperimentConfig(experiment_name="test")
    c1.model.type = "invalid_type"
    with pytest.raises(ValueError, match="Model type must be"):
        c1.validate()

    # Invalid scale
    c2 = ExperimentConfig(experiment_name="test")
    c2.model.scale = 3
    with pytest.raises(ValueError, match="Model scale .* not supported"):
        c2.validate()

    # Invalid dataset
    c3 = ExperimentConfig(experiment_name="test")
    c3.dataset.name = "UnknownDataset"
    with pytest.raises(ValueError, match="Dataset name .* must be one of"):
        c3.validate()


def test_checkpoint_manager() -> None:
    """Verify saving, searching, and loading checkpoints."""
    state_dict = {
        "epoch": 25,
        "model_weights": {"weight1": torch.tensor([1.0, 2.0])},
        "optimizer_state": {"lr": 1e-4}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_25.pth"

        # 1. Save checkpoint
        CheckpointManager.save_checkpoint(state_dict, checkpoint_path)
        assert checkpoint_path.is_file()

        # 2. Search latest
        latest = CheckpointManager.get_latest_checkpoint(checkpoint_dir)
        assert latest == checkpoint_path

        # 3. Resume training loading
        res_state, start_epoch = CheckpointManager.resume_training(checkpoint_dir)
        assert start_epoch == 26
        assert res_state is not None
        assert res_state["epoch"] == 25
        assert torch.equal(res_state["model_weights"]["weight1"], state_dict["model_weights"]["weight1"])
