"""
Unit tests validating FeatureExtractor imports, instantiation, configuration, parameter counts, and forward passes.
"""

# pyrefly: ignore [missing-import]
import pytest
import torch
import logging
from training.models.modules.feature_extractor import FeatureExtractor
from training.utils.config import ExperimentConfig, ModelConfig
from training.models.utils import count_parameters


def test_feature_extractor_init() -> None:
    """Verify FeatureExtractor properties initialization."""
    extractor = FeatureExtractor(in_channels=3, out_channels=48)
    assert extractor.in_channels == 3
    assert extractor.out_channels == 48
    assert extractor.kernel_size == 3
    assert extractor.padding == 1
    assert extractor.stride == 1
    assert extractor.bias is True
    assert extractor.negative_slope == 0.05
    assert isinstance(extractor, torch.nn.Module)


def test_feature_extractor_config_loading() -> None:
    """Verify FeatureExtractor configuration loading with custom settings."""
    model_cfg = ModelConfig(
        in_channels=3,
        num_channels=48,
        kernel_size=5,
        padding=2,
        stride=2,
        bias=False,
        negative_slope=0.1
    )
    # Instantiate with explicit parameters extracted from ModelConfig
    extractor = FeatureExtractor(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.num_channels,
        kernel_size=model_cfg.kernel_size,
        padding=model_cfg.padding,
        stride=model_cfg.stride,
        bias=model_cfg.bias,
        negative_slope=model_cfg.negative_slope,
    )
    assert extractor.in_channels == 3
    assert extractor.out_channels == 48
    assert extractor.kernel_size == 5
    assert extractor.padding == 2
    assert extractor.stride == 2
    assert extractor.bias is False
    assert extractor.negative_slope == 0.1

    # Instantiate with dict config parameters passed explicitly
    dict_cfg = {
        "in_channels": 4,
        "num_channels": 64,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "bias": True,
        "negative_slope": 0.01
    }
    extractor_dict = FeatureExtractor(
        in_channels=dict_cfg["in_channels"],
        out_channels=dict_cfg["num_channels"],
        kernel_size=dict_cfg["kernel_size"],
        padding=dict_cfg["padding"],
        stride=dict_cfg["stride"],
        bias=dict_cfg["bias"],
        negative_slope=dict_cfg["negative_slope"],
    )
    assert extractor_dict.in_channels == 4
    assert extractor_dict.out_channels == 64
    assert extractor_dict.kernel_size == 3
    assert extractor_dict.padding == 1
    assert extractor_dict.stride == 1
    assert extractor_dict.bias is True
    assert extractor_dict.negative_slope == 0.01


def test_feature_extractor_dynamic_parameter_count() -> None:
    """Verify FeatureExtractor parameter counts match dynamic calculations."""
    for in_c, out_c, k_size, use_bias in [
        (3, 48, 3, True),
        (3, 32, 3, False),
        (4, 64, 5, True),
    ]:
        extractor = FeatureExtractor(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k_size,
            bias=use_bias
        )
        # Compute expected parameter count dynamically
        expected_params = (in_c * out_c * k_size * k_size) + (out_c if use_bias else 0)
        actual_params = count_parameters(extractor)
        assert actual_params == expected_params, (
            f"Param count mismatch. Expected {expected_params}, got {actual_params}"
        )


def test_feature_extractor_forward_various_shapes() -> None:
    """Verify FeatureExtractor forward pass across different shape configurations."""
    # Test cases: (Batch, Channels, Height, Width)
    test_shapes = [
        (2, 3, 48, 48),  # Square shape
        (1, 3, 64, 48),  # Batch 1, non-square shape
        (4, 3, 32, 64),  # Larger batch, non-square shape
    ]
    extractor = FeatureExtractor(in_channels=3, out_channels=48)

    for shape in test_shapes:
        dummy_input = torch.randn(shape)
        with torch.no_grad():
            output = extractor(dummy_input)

        # Spatial dimensions H, W should remain unchanged, channels should match out_channels
        expected_shape = [shape[0], 48, shape[2], shape[3]]
        assert list(output.shape) == expected_shape
        assert output.dtype == dummy_input.dtype
        assert output.device == dummy_input.device


def test_feature_extractor_dtypes() -> None:
    """Verify FeatureExtractor preserves and operates correctly under different floating point dtypes."""
    extractor = FeatureExtractor(in_channels=3, out_channels=32)

    for dtype in [torch.float32, torch.float64]:
        dummy_input = torch.randn(2, 3, 48, 48, dtype=dtype)
        
        # Transfer module parameters to matching dtype
        extractor.to(dtype)
        with torch.no_grad():
            output = extractor(dummy_input)

        assert output.dtype == dtype
        assert list(output.shape) == [2, 32, 48, 48]


def test_feature_extractor_invalid_inputs() -> None:
    """Verify FeatureExtractor raises informative errors on invalid input types and shapes."""
    extractor = FeatureExtractor(in_channels=3, out_channels=32)

    # 1. Non-tensor input
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        extractor("not a tensor")

    # 2. Incorrect dimensionality (not 4D)
    with pytest.raises(ValueError, match="must have 4 dimensions"):
        extractor(torch.randn(3, 48, 48))  # 3D tensor

    # 3. Channels mismatch
    with pytest.raises(ValueError, match="channel dimension mismatch"):
        extractor(torch.randn(2, 4, 48, 48))  # 4 channels instead of 3


def test_feature_extractor_debug_mode(caplog: pytest.LogCaptureFixture) -> None:
    """Verify FeatureExtractor logs shape information when debug is enabled."""
    extractor = FeatureExtractor(in_channels=3, out_channels=32, debug=True)
    dummy_input = torch.randn(2, 3, 48, 48)

    with caplog.at_level(logging.INFO):
        with torch.no_grad():
            _ = extractor(dummy_input)

    # Check that debug log outputs contain expected shape and device keys
    assert any("Input Shape" in record.message for record in caplog.records)
    assert any("Output Shape" in record.message for record in caplog.records)
    assert any("Current Device" in record.message for record in caplog.records)
    assert any("Tensor Dtype" in record.message for record in caplog.records)


def test_feature_extractor_extra_repr() -> None:
    """Verify extra_repr() contains configuration parameters."""
    extractor = FeatureExtractor(in_channels=3, out_channels=32, kernel_size=3, negative_slope=0.05)
    repr_str = repr(extractor)
    assert "in_channels=3" in repr_str
    assert "out_channels=32" in repr_str
    assert "negative_slope=0.05" in repr_str


def test_feature_extractor_device_execution() -> None:
    """Verify FeatureExtractor runs on CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    extractor = FeatureExtractor(in_channels=3, out_channels=32).cuda()
    dummy_input = torch.randn(2, 3, 48, 48).cuda()

    with torch.no_grad():
        output = extractor(dummy_input)

    assert output.is_cuda
    assert list(output.shape) == [2, 32, 48, 48]
