"""
Unit tests validating ResidualFeatureDistillationBlock configurations, parameter counts,
stage-wise functionality, alpha gradient flow, serialization, and trace exports.
"""

import io
import pytest
import torch
import logging
from training.models.modules.rfdb import ResidualFeatureDistillationBlock
from training.utils.config import ModelConfig
from training.models.utils import count_parameters


def test_distillation_block_init() -> None:
    """Verify ResidualFeatureDistillationBlock properties."""
    # Test main class
    block = ResidualFeatureDistillationBlock(channels=48, distilled_channels=24)
    assert block.num_channels == 48
    assert block.distilled_channels == 24
    assert block.kernel_size == 3
    assert block.padding == 1
    assert block.stride == 1
    assert block.bias is True
    assert block.negative_slope == 0.05
    assert block.initial_residual_scale == 1.0
    assert block.block_index is None
    assert block.debug is False
    assert isinstance(block, torch.nn.Module)


def test_distillation_block_config_loading() -> None:
    """Verify ResidualFeatureDistillationBlock config loading with custom objects."""
    model_cfg = ModelConfig(
        num_blocks=4,
        num_channels=64,
        distilled_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=False,
        negative_slope=0.1,
        initial_residual_scale=0.5
    )
    # Instantiate with explicit parameters extracted from ModelConfig
    block = ResidualFeatureDistillationBlock(
        channels=model_cfg.num_channels,
        distilled_channels=model_cfg.distilled_channels,
        kernel_size=model_cfg.kernel_size,
        padding=model_cfg.padding,
        stride=model_cfg.stride,
        bias=model_cfg.bias,
        negative_slope=model_cfg.negative_slope,
        initial_residual_scale=model_cfg.initial_residual_scale,
        block_index=2,
        debug=True,
    )
    assert block.num_channels == 64
    assert block.distilled_channels == 32
    assert block.bias is False
    assert block.negative_slope == 0.1
    assert block.initial_residual_scale == 0.5
    assert block.block_index == 2
    assert block.debug is True

    # Instantiate with dict config parameters passed explicitly
    dict_cfg = {
        "num_channels": 32,
        "distilled_channels": 16,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "bias": True,
        "negative_slope": 0.01,
        "initial_residual_scale": 0.2
    }
    block_dict = ResidualFeatureDistillationBlock(
        channels=dict_cfg["num_channels"],
        distilled_channels=dict_cfg["distilled_channels"],
        kernel_size=dict_cfg["kernel_size"],
        padding=dict_cfg["padding"],
        stride=dict_cfg["stride"],
        bias=dict_cfg["bias"],
        negative_slope=dict_cfg["negative_slope"],
        initial_residual_scale=dict_cfg["initial_residual_scale"],
    )
    assert block_dict.num_channels == 32
    assert block_dict.distilled_channels == 16
    assert block_dict.initial_residual_scale == 0.2


def test_distillation_block_input_validation() -> None:
    """Verify input checks correctly throw errors on invalid input data structures."""
    block = ResidualFeatureDistillationBlock(channels=48)

    # 1. Non-tensor input
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        block("invalid input")

    # 2. Incorrect dimensionality (not 4D)
    with pytest.raises(ValueError, match="must have 4 dimensions"):
        block(torch.randn(3, 48, 48))

    # 3. Channels mismatch
    with pytest.raises(ValueError, match="channel dimension mismatch"):
        block(torch.randn(2, 32, 48, 48))


def test_distillation_block_forward_various_shapes() -> None:
    """Verify spatial and channel-wise shape preservation across batch size and dimension variations."""
    test_shapes = [
        (2, 48, 48, 48),  # Standard batch
        (1, 48, 64, 48),  # Batch size 1, non-square image
        (4, 48, 32, 64),  # Larger batch, non-square image
    ]
    block = ResidualFeatureDistillationBlock(channels=48, distilled_channels=24)

    for shape in test_shapes:
        dummy_input = torch.randn(shape)
        with torch.no_grad():
            output = block(dummy_input)

        # Output shape must exactly match input shape
        assert list(output.shape) == list(shape)
        assert output.dtype == dummy_input.dtype
        assert output.device == dummy_input.device


def test_distillation_block_stages_and_splits() -> None:
    """Verify channel selections and stage splits function according to mathematical expectations."""
    block = ResidualFeatureDistillationBlock(channels=48, distilled_channels=16)
    dummy_input = torch.randn(2, 48, 24, 24)

    # 1. Feature refinement output shape
    refined = block._feature_refinement(dummy_input)
    assert refined.shape == (2, 48, 24, 24)

    # 2. Information Extraction Branch (takes entire refined tensor: 48 channels)
    distilled_out = block._information_extraction_branch(refined)
    assert distilled_out.shape == (2, 16, 24, 24)

    # 3. Feature Preservation Branch (performs identity slice)
    # Preservation branch must slice refined tensor from index distilled_channels to num_channels
    preserved_out = block._feature_preservation_branch(refined)
    assert preserved_out.shape == (2, 32, 24, 24)
    assert torch.equal(preserved_out, refined[:, 16:, :, :])

    # 4. Concatenation and fusion shapes
    concatenated = torch.cat([distilled_out, preserved_out], dim=1)
    assert concatenated.shape == (2, 48, 24, 24)

    fused = block._feature_fusion(concatenated)
    assert fused.shape == (2, 48, 24, 24)


def test_distillation_block_learnable_alpha() -> None:
    """Verify alpha exists, has channel-wise shape [1, C, 1, 1], and requires grad."""
    block = ResidualFeatureDistillationBlock(channels=48, initial_residual_scale=0.2)
    assert hasattr(block, "alpha")
    assert isinstance(block.alpha, torch.nn.Parameter)
    assert list(block.alpha.shape) == [1, 48, 1, 1]
    assert block.alpha.requires_grad is True

    # Initial values should match initial scale
    assert torch.allclose(block.alpha, torch.full((1, 48, 1, 1), 0.2))


def test_distillation_block_gradient_flow() -> None:
    """Verify gradient flow through learnable alpha during a backward step."""
    block = ResidualFeatureDistillationBlock(channels=48, initial_residual_scale=0.2)
    dummy_input = torch.randn(2, 48, 24, 24)

    # Clear gradients
    if block.alpha.grad is not None:
        block.alpha.grad.zero_()

    # Forward, loss, backward
    output = block(dummy_input)
    loss = output.sum()
    loss.backward()

    # Verify gradients computed for alpha
    assert block.alpha.grad is not None
    assert torch.any(block.alpha.grad != 0.0)


def test_distillation_block_serialization() -> None:
    """Verify serialization and state_dict reloading for alpha parameter."""
    block = ResidualFeatureDistillationBlock(channels=48, initial_residual_scale=0.5)
    
    # Modify alpha parameter to verify loading
    with torch.no_grad():
        block.alpha.fill_(0.9)

    # Serialize block parameters
    buffer = io.BytesIO()
    torch.save(block.state_dict(), buffer)
    buffer.seek(0)

    # Load into a new block
    new_block = ResidualFeatureDistillationBlock(channels=48, initial_residual_scale=0.5)
    assert not torch.allclose(new_block.alpha, block.alpha) # Verify difference before loading

    new_block.load_state_dict(torch.load(buffer))
    assert torch.allclose(new_block.alpha, block.alpha) # Verify loaded value
    assert new_block.alpha.requires_grad is True


def test_distillation_block_residual_equation() -> None:
    """Verify forward matches the residual equation: Output = Input + alpha * FusionOutput."""
    block = ResidualFeatureDistillationBlock(channels=48, initial_residual_scale=0.5)
    dummy_input = torch.randn(2, 48, 24, 24)

    with torch.no_grad():
        # Step-by-step emulation
        refined = block._feature_refinement(dummy_input)
        d_out = block._information_extraction_branch(refined)
        p_out = block._feature_preservation_branch(refined)
        concat = torch.cat([d_out, p_out], dim=1)
        fused = block._feature_fusion(concat)
        expected_output = dummy_input + (block.alpha * fused)

        # Forward output comparison
        forward_output = block(dummy_input)
        assert torch.allclose(forward_output, expected_output, atol=1e-6)


def test_distillation_block_dtypes() -> None:
    """Verify block preserves data types for float32 and float64 dtypes."""
    block = ResidualFeatureDistillationBlock(channels=32)

    for dtype in [torch.float32, torch.float64]:
        dummy_input = torch.randn(2, 32, 24, 24, dtype=dtype)
        block.to(dtype)

        with torch.no_grad():
            output = block(dummy_input)

        assert output.dtype == dtype
        assert list(output.shape) == [2, 32, 24, 24]


def test_distillation_block_torchscript() -> None:
    """Verify TorchScript tracing and script export compatibility."""
    block = ResidualFeatureDistillationBlock(channels=32)
    dummy_input = torch.randn(2, 32, 24, 24)

    # Trace
    try:
        traced_block = torch.jit.trace(block, dummy_input)
        traced_output = traced_block(dummy_input)
        assert list(traced_output.shape) == [2, 32, 24, 24]
    except Exception as e:
        pytest.fail(f"TorchScript tracing failed: {e}")

    # Script
    try:
        scripted_block = torch.jit.script(block)
        scripted_output = scripted_block(dummy_input)
        assert list(scripted_output.shape) == [2, 32, 24, 24]
    except Exception as e:
        pytest.fail(f"TorchScript scripting failed: {e}")


def test_distillation_block_dynamic_parameters() -> None:
    """Verify block parameter counts dynamically match architecture equations."""
    for C, D, k, use_bias in [
        (48, 24, 3, True),
        (32, 16, 3, False),
        (64, 16, 5, True),
    ]:
        block = ResidualFeatureDistillationBlock(
            channels=C,
            distilled_channels=D,
            kernel_size=k,
            bias=use_bias
        )
        
        # Expected parameters per stage under finalized architecture
        refinement_params = (C * C * k * k) + (C if use_bias else 0)
        # Note: Information extraction maps C -> D (C * D * 1 * 1) + D
        extraction_params = (C * D * 1 * 1) + (D if use_bias else 0)
        preservation_params = 0
        fusion_params = (C * C * 1 * 1) + (C if use_bias else 0)
        alpha_params = C # broadcast shape parameter of shape (1, C, 1, 1)

        expected_total = refinement_params + extraction_params + preservation_params + fusion_params + alpha_params
        actual_total = count_parameters(block)
        assert actual_total == expected_total, (
            f"Param mismatch for channels={C}. Expected {expected_total}, got {actual_total}"
        )


def test_distillation_block_debug_mode(caplog: pytest.LogCaptureFixture) -> None:
    """Verify block logs stage-wise shape data in debug mode and checks shape assertions."""
    block = ResidualFeatureDistillationBlock(channels=48, distilled_channels=24, block_index=5, debug=True)
    dummy_input = torch.randn(2, 48, 24, 24)

    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            _ = block(dummy_input)

    log_msgs = [record.message for record in caplog.records]
    assert any("RFDB Block 5 Debug Pass" in msg for msg in log_msgs)
    assert any("Refined Shape" in msg for msg in log_msgs)
    assert any("Distilled Shape" in msg for msg in log_msgs)
    assert any("Preserved Shape" in msg for msg in log_msgs)
    assert any("Concatenated Shape" in msg for msg in log_msgs)
    assert any("Fusion Shape" in msg for msg in log_msgs)
    assert any("Alpha Tensor Shape" in msg for msg in log_msgs)
    assert any("Alpha Mean Value" in msg for msg in log_msgs)


def test_distillation_block_device_execution() -> None:
    """Verify block runs on CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    block = ResidualFeatureDistillationBlock(channels=32).cuda()
    dummy_input = torch.randn(2, 32, 24, 24).cuda()

    with torch.no_grad():
        output = block(dummy_input)

    assert output.is_cuda
    assert list(output.shape) == [2, 32, 24, 24]
