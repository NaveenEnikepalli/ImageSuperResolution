"""Comprehensive Inference Pipeline and Model Integration Tests.

Author: Antigravity
Purpose: Validate Student x4, Student x2, SwinIR x8, Model Selector routing,
CPU inference execution, timing, and output dimensions.
"""

import sys
import time
import pytest
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference.student_inference import StudentInference
from inference.swinir_inference import SwinIRInference
from inference.model_selector import select_model
from training.models.student_model import StudentModel
from training.utils.config import load_config


class TestStudentX4Inference:
    """Test suite for Student x4 Model and Inference."""

    def test_student_x4_architecture(self):
        """Verify Student x4 model initialization and forward pass on CPU."""
        model = StudentModel(scale=4)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_student_x4_cpu_inference(self):
        """Verify Student x4 CPU inference pipeline and output image dimensions."""
        engine = StudentInference(scale=4, device="cpu")
        img = Image.new("RGB", (64, 48), color=(100, 150, 200))

        start = time.time()
        out_img = engine.enhance(img)
        elapsed = time.time() - start

        assert out_img.size == (256, 192)
        assert elapsed < 5.0, f"Student x4 CPU inference took too long: {elapsed:.2f}s"


class TestStudentX2Inference:
    """Test suite for Student x2 Model, Training Config, and Inference."""

    def test_student_x2_config(self):
        """Verify Student x2 training configuration file."""
        config_path = Path("training/configs/student_x2_training.yaml")
        assert config_path.exists()
        config = load_config(config_path)
        assert config.model.scale == 2
        assert config.model.num_channels == 48

    def test_student_x2_architecture(self):
        """Verify Student x2 model architecture instantiated with scale=2."""
        model = StudentModel(scale=2)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_student_x2_cpu_inference(self):
        """Verify Student x2 CPU inference pipeline and output dimensions."""
        engine = StudentInference(scale=2, device="cpu")
        img = Image.new("RGB", (80, 60), color=(50, 100, 150))

        start = time.time()
        out_img = engine.enhance(img)
        elapsed = time.time() - start

        assert out_img.size == (160, 120)
        assert elapsed < 5.0, f"Student x2 CPU inference took too long: {elapsed:.2f}s"


class TestSwinIRX8Inference:
    """Test suite for SwinIR x8 Model and Inference."""

    def test_swinir_x8_cpu_inference(self):
        """Verify SwinIR x8 CPU inference, window padding, timing, and output dimensions."""
        # Use small input image to test CPU inference safely
        engine = SwinIRInference(scale=8, device="cpu", download_if_missing=False)
        img = Image.new("RGB", (16, 16), color=(200, 100, 50))

        start = time.time()
        out_img = engine.enhance(img)
        elapsed = time.time() - start

        assert out_img.size == (128, 128)
        print(f"\n[Timing Report] SwinIR x8 CPU inference on 16x16 input image: {elapsed:.4f} seconds")


class TestModelSelector:
    """Test suite for automatic model selection and routing."""

    def test_select_scale_2(self):
        """Verify scale=2 routes to StudentInference with scale=2."""
        engine = select_model(2, device="cpu")
        assert isinstance(engine, StudentInference)
        assert engine.scale == 2

    def test_select_scale_4(self):
        """Verify scale=4 routes to StudentInference with scale=4."""
        engine = select_model(4, device="cpu")
        assert isinstance(engine, StudentInference)
        assert engine.scale == 4

    def test_select_scale_8(self):
        """Verify scale=8 routes to SwinIRInference with scale=8."""
        engine = select_model(8, device="cpu")
        assert isinstance(engine, SwinIRInference)
        assert engine.scale == 8

    def test_invalid_scale_raises_error(self):
        """Verify invalid scale values raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported super-resolution scale factor"):
            select_model(3)

        with pytest.raises(ValueError, match="Unsupported super-resolution scale factor"):
            select_model(5)
