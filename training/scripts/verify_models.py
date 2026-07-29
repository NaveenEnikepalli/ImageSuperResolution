"""
Diagnostic verification script validating Phase 4A model structures, wrappers, utilities, and configurations.
"""

import sys
import logging
from pathlib import Path
import torch

from training.utils.config import load_config
from training.utils.logger import setup_logger
from training.models.student_model import StudentModel
from training.models.modules.feature_extractor import FeatureExtractor
from training.models.modules.rfdb import ResidualFeatureDistillationBlock
from training.teacher import SwinIRWrapper
from training.models.utils import (
    count_parameters,
    count_trainable_parameters,
    estimate_model_size,
    print_model_summary,
    verify_forward_signature,
    verify_device,
    verify_dtype,
    custom_initialize
)

logger = logging.getLogger("VerifyModels")


def verify_folder_structure(base_path: Path) -> bool:
    """Check that all required Phase 4A folder paths exist.

    Args:
        base_path (Path): Path to training/ directory.

    Returns:
        bool: True if structure is correct.
    """
    required_paths = [
        base_path / "models" / "modules",
        base_path / "teacher",
        base_path / "losses",
        base_path / "models" / "utils",
        base_path / "experiments",
        base_path / "tests" / "models"
    ]
    
    success = True
    logger.info("Checking repository folder structure:")
    for path in required_paths:
        exists = path.is_dir()
        logger.info(f"  {path.relative_to(base_path.parent)}: {'Found' if exists else 'MISSING'}")
        if not exists:
            success = False
    return success


def run_model_diagnostics(config_path: Path) -> bool:
    """Instantiate mock architectures and run utility validations.

    Args:
        config_path (Path): Path to YAML config template.

    Returns:
        bool: True if check results are correct.
    """
    logger.info("Loading default configuration settings...")
    config = load_config(str(config_path))
    
    # 1. Standalone FeatureExtractor diagnostics
    logger.info("Initializing FeatureExtractor standalone with config...")
    m_cfg = config.model
    extractor = FeatureExtractor(
        in_channels=getattr(m_cfg, "in_channels", 3),
        out_channels=getattr(m_cfg, "num_channels", 48),
        kernel_size=getattr(m_cfg, "kernel_size", 3),
        padding=getattr(m_cfg, "padding", 1),
        stride=getattr(m_cfg, "stride", 1),
        bias=getattr(m_cfg, "bias", True),
        negative_slope=getattr(m_cfg, "negative_slope", 0.05),
    )
    
    # Check parameters and sizing
    fe_in_channels = extractor.in_channels
    fe_out_channels = extractor.out_channels
    fe_kernel_size = extractor.kernel_size
    fe_bias = extractor.bias
    
    # Compute expected parameters dynamically
    expected_fe_params = (fe_in_channels * fe_out_channels * fe_kernel_size * fe_kernel_size) + (fe_out_channels if fe_bias else 0)
    actual_fe_params = count_parameters(extractor)
    fe_trainable_params = count_trainable_parameters(extractor)
    fe_size_mb = estimate_model_size(extractor)
    
    logger.info(f"  FeatureExtractor input shape expected:  [B, {fe_in_channels}, H, W]")
    logger.info(f"  FeatureExtractor output shape expected: [B, {fe_out_channels}, H, W]")
    logger.info(f"  FeatureExtractor expected params:       {expected_fe_params}")
    logger.info(f"  FeatureExtractor total params:          {actual_fe_params}")
    logger.info(f"  FeatureExtractor trainable params:      {fe_trainable_params}")
    logger.info(f"  FeatureExtractor estimated size:        {fe_size_mb:.6f} MB")
    
    assert actual_fe_params == expected_fe_params, f"FeatureExtractor parameters mismatch: expected {expected_fe_params}, got {actual_fe_params}"
    
    # Verify forward signature
    fe_input_shape = (2, fe_in_channels, 48, 48)
    logger.info(f"  Verifying FeatureExtractor forward signature with shape {fe_input_shape}...")
    fe_forward_ok = verify_forward_signature(extractor, fe_input_shape)
    logger.info(f"  FeatureExtractor forward status:        {'SUCCESS' if fe_forward_ok else 'FAILED'}")
    assert fe_forward_ok, "FeatureExtractor forward signature check failed"
    
    # Print standalone summary
    print_model_summary(extractor, fe_input_shape)

    # 1.3. Standalone ResidualFeatureDistillationBlock diagnostics
    logger.info("Initializing ResidualFeatureDistillationBlock standalone with config...")
    rfdb = ResidualFeatureDistillationBlock(
        channels=getattr(m_cfg, "num_channels", 48),
        distilled_channels=getattr(m_cfg, "distilled_channels", 24),
        kernel_size=getattr(m_cfg, "kernel_size", 3),
        padding=getattr(m_cfg, "padding", 1),
        stride=getattr(m_cfg, "stride", 1),
        bias=getattr(m_cfg, "bias", True),
        negative_slope=getattr(m_cfg, "negative_slope", 0.05),
        initial_residual_scale=getattr(m_cfg, "initial_residual_scale", 1.0),
        block_index=1,
    )
    
    # Check parameters and sizing
    rfdb_channels = rfdb.num_channels
    rfdb_distilled = rfdb.distilled_channels
    rfdb_k = rfdb.kernel_size
    rfdb_bias = rfdb.bias
    
    # Calculate parameter breakdown
    refinement_params = (rfdb_channels * rfdb_channels * rfdb_k * rfdb_k) + (rfdb_channels if rfdb_bias else 0)
    extraction_params = (rfdb_channels * rfdb_distilled * 1 * 1) + (rfdb_distilled if rfdb_bias else 0)
    preservation_params = 0
    fusion_params = (rfdb_channels * rfdb_channels * 1 * 1) + (rfdb_channels if rfdb_bias else 0)
    alpha_params = rfdb_channels  # shape (1, num_channels, 1, 1)
    
    expected_rfdb_params = refinement_params + extraction_params + preservation_params + fusion_params + alpha_params
    actual_rfdb_params = count_parameters(rfdb)
    rfdb_trainable_params = count_trainable_parameters(rfdb)
    rfdb_size_mb = estimate_model_size(rfdb)
    
    logger.info(f"  RFDB Stage 1 - Feature Refinement Params:    {refinement_params}")
    logger.info(f"  RFDB Stage 2 - Information Extraction:       {extraction_params}")
    logger.info(f"  RFDB Stage 3 - Feature Preservation:         {preservation_params}")
    logger.info(f"  RFDB Stage 4 - Feature Fusion:               {fusion_params}")
    logger.info(f"  RFDB Stage 5 - Channel-wise Residual Scale:  {alpha_params}")
    logger.info(f"  RFDB expected parameters (total):            {expected_rfdb_params}")
    logger.info(f"  RFDB total parameters (actual):              {actual_rfdb_params}")
    logger.info(f"  RFDB trainable parameters:                   {rfdb_trainable_params}")
    logger.info(f"  RFDB estimated size:                         {rfdb_size_mb:.6f} MB")
    logger.info(f"  RFDB Alpha Parameter Shape:                  {list(rfdb.alpha.shape)}")
    logger.info(f"  RFDB Alpha Mean Value:                       {rfdb.alpha.mean().item():.6f}")
    
    assert actual_rfdb_params == expected_rfdb_params, f"RFDB parameters mismatch: expected {expected_rfdb_params}, got {actual_rfdb_params}"
    
    # Verify forward signature
    rfdb_input_shape = (2, rfdb_channels, 48, 48)
    logger.info(f"  Verifying RFDB forward signature with shape {rfdb_input_shape}...")
    rfdb_forward_ok = verify_forward_signature(rfdb, rfdb_input_shape)
    logger.info(f"  RFDB forward status:                        {'SUCCESS' if rfdb_forward_ok else 'FAILED'}")
    assert rfdb_forward_ok, "RFDB forward signature check failed"
    
    # Print standalone summary
    print_model_summary(rfdb, rfdb_input_shape)

    # 1.5. Instantiate Student Model
    logger.info("Initializing StudentModel with config...")
    student = StudentModel(
        in_channels=getattr(m_cfg, "in_channels", 3),
        out_channels=3,
        num_features=getattr(m_cfg, "num_channels", 48),
        distilled_channels=getattr(m_cfg, "distilled_channels", 24),
        num_blocks=getattr(m_cfg, "num_blocks", 3),
        scale=getattr(m_cfg, "scale", 4),
        negative_slope=getattr(m_cfg, "negative_slope", 0.05),
    )
    
    # Check parameters and sizing
    total_params = count_parameters(student)
    trainable_params = count_trainable_parameters(student)
    size_mb = estimate_model_size(student)
    
    logger.info(f"  Student total params:     {total_params}")
    logger.info(f"  Student trainable params: {trainable_params}")
    logger.info(f"  Student estimated size:   {size_mb:.6f} MB")
    
    # Verify forward signature
    input_shape = (2, 3, 48, 48)  # Batch 2, RGB, 48x48
    logger.info(f"  Verifying forward signature with shape {input_shape}...")
    forward_ok = verify_forward_signature(student, input_shape)
    logger.info(f"  Forward execution status: {'SUCCESS' if forward_ok else 'FAILED'}")
    assert forward_ok, "Student forward signature check failed"

    # Print model summary
    print_model_summary(student, input_shape)

    # Apply mock custom weights initialization
    logger.info("Applying Kaiming Normal initialization checks...")
    custom_initialize(student, method="kaiming", nonlinearity="relu")
    logger.info("  ✓ Initialization diagnostics executed successfully.")

    # 2. Instantiate Teacher SwinIR Wrapper
    logger.info("Initializing SwinIRWrapper teacher model...")
    teacher = SwinIRWrapper(scale=config.model.scale)
    
    # Check that parameters requires_grad has successfully turned False (frozen by default)
    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher param {name} was not frozen!"
        
    assert not teacher.training, "Teacher model is not in evaluation mode!"
    logger.info("  ✓ Teacher wrapper parameters frozen and in eval mode verified.")

    # Verify device and data type properties
    assert verify_device(student, torch.device("cpu")), "Student device check failed"
    assert verify_dtype(student, torch.float32), "Student dtype check failed"
    logger.info("  ✓ Device and dtype checks passed.")

    return True


def main() -> None:
    setup_logger(name="training", level=logging.INFO)
    
    # Resolve paths
    scripts_dir = Path(__file__).resolve().parent
    training_dir = scripts_dir.parent
    
    config_path = training_dir / "configs" / "default_config.yaml"
    
    logger.info("=" * 60)
    logger.info("PHASE 4A MODEL FRAMEWORK VERIFICATION")
    logger.info("=" * 60)
    
    # Run checks
    structure_ok = verify_folder_structure(training_dir)
    if not structure_ok:
        logger.error("Verification failed due to missing directory path structures.")
        sys.exit(1)
        
    try:
        diagnostics_ok = run_model_diagnostics(config_path)
    except Exception as e:
        logger.error(f"Verification failed during models diagnostics. Error: {e}")
        sys.exit(1)
        
    if structure_ok and diagnostics_ok:
        logger.info("=" * 60)
        logger.info("PHASE 4A MODEL FRAMEWORK VERIFICATION: SUCCESS")
        logger.info("All templates, custom wrappers, and diagnostic utilities verify.")
        logger.info("=" * 60)
    else:
        logger.error("Verification completed with diagnostic errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
