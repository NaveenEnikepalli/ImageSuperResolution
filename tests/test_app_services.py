"""Unit and Integration Tests for Application Services, Routing, and Preprocessing.

Author: Antigravity
Purpose: Verify model routing (2x->Student x2, 4x->Student x4, 8x->SwinIR x8), invalid scale handling,
image validation, preprocessing, output scaling dimensions, and caching behavior.
"""

import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from backend.app.config import MAX_UPLOAD_SIZE_MB
from backend.app.services.image_service import (
    validate_image_file,
    load_and_preprocess_image,
    encode_image_to_png,
    get_image_info,
)
from backend.app.services.model_service import get_model_engine, clear_model_cache
from backend.app.services.enhancement_service import enhance_image
from inference.model_selector import select_model, SUPPORTED_SCALES
from inference.student_inference import StudentInference
from inference.swinir_inference import SwinIRInference


# Helper function to generate test PNG image bytes
def create_dummy_image_bytes(width: int = 32, height: int = 24, format: str = "PNG") -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestModelRouting:
    """Tests for model selection and routing rules."""

    def test_supported_scales_constant(self):
        assert set(SUPPORTED_SCALES) == {2, 4, 8}

    @patch("inference.model_selector.StudentInference")
    def test_route_scale_2_to_student_x2(self, mock_student):
        mock_instance = MagicMock()
        mock_student.return_value = mock_instance

        engine = select_model(scale=2)
        mock_student.assert_called_once_with(scale=2, checkpoint_path=None, device=None)
        assert engine == mock_instance

    @patch("inference.model_selector.StudentInference")
    def test_route_scale_4_to_student_x4(self, mock_student):
        mock_instance = MagicMock()
        mock_student.return_value = mock_instance

        engine = select_model(scale=4)
        mock_student.assert_called_once_with(scale=4, checkpoint_path=None, device=None)
        assert engine == mock_instance

    @patch("inference.model_selector.SwinIRInference")
    def test_route_scale_8_to_swinir_x8(self, mock_swinir):
        mock_instance = MagicMock()
        mock_swinir.return_value = mock_instance

        engine = select_model(scale=8)
        mock_swinir.assert_called_once_with(scale=8, checkpoint_path=None, device=None)
        assert engine == mock_instance

    @pytest.mark.parametrize("invalid_scale", [1, 3, 5, 6, 10, -2, 0])
    def test_invalid_scale_raises_value_error(self, invalid_scale):
        with pytest.raises(ValueError) as exc_info:
            select_model(scale=invalid_scale)
        assert f"x{invalid_scale}" in str(exc_info.value) or "Unsupported" in str(exc_info.value)


class TestImageValidationAndPreprocessing:
    """Tests for image file validation, format verification, and preprocessing."""

    def test_valid_png_image(self):
        img_bytes = create_dummy_image_bytes(64, 48, format="PNG")
        is_valid, err = validate_image_file(img_bytes, filename="test.png")
        assert is_valid is True
        assert err is None

    def test_valid_jpeg_image(self):
        img_bytes = create_dummy_image_bytes(64, 48, format="JPEG")
        is_valid, err = validate_image_file(img_bytes, filename="test.jpg")
        assert is_valid is True
        assert err is None

    def test_empty_bytes_rejected(self):
        is_valid, err = validate_image_file(b"", filename="empty.png")
        assert is_valid is False
        assert "empty" in err.lower()

    def test_corrupt_bytes_rejected(self):
        is_valid, err = validate_image_file(b"NOT_AN_IMAGE_DATA_12345", filename="corrupt.png")
        assert is_valid is False
        assert "invalid" in err.lower() or "corrupted" in err.lower()

    def test_oversized_file_rejected(self):
        oversized_bytes = b"0" * ((MAX_UPLOAD_SIZE_MB * 1024 * 1024) + 100)
        is_valid, err = validate_image_file(oversized_bytes, filename="huge.png")
        assert is_valid is False
        assert "exceeds" in err.lower()

    def test_load_and_preprocess_converts_to_rgb(self):
        rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        rgba_img.save(buf, format="PNG")

        rgb_result = load_and_preprocess_image(buf.getvalue())
        assert rgb_result.mode == "RGB"
        assert rgb_result.size == (32, 32)


class TestModelServiceCaching:
    """Tests for model caching behavior."""

    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    @patch("backend.app.services.model_service.select_model")
    def test_get_model_engine_caches_instance(self, mock_select_model):
        mock_engine = MagicMock()
        mock_select_model.return_value = mock_engine

        engine1 = get_model_engine(4)
        engine2 = get_model_engine(4)

        assert engine1 is mock_engine
        assert engine2 is mock_engine
        mock_select_model.assert_called_once_with(scale=4, device=None)


class TestEnhancementServiceAndScaling:
    """Tests for complete enhancement pipeline execution and dimension scaling."""

    @patch("backend.app.services.enhancement_service.get_model_engine")
    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_output_resolution_scaling(self, mock_get_engine, scale):
        in_w, in_h = 20, 15
        img_bytes = create_dummy_image_bytes(in_w, in_h, format="PNG")

        out_w, out_h = in_w * scale, in_h * scale
        out_pil = Image.fromarray(np.zeros((out_h, out_w, 3), dtype=np.uint8), mode="RGB")

        mock_engine = MagicMock()
        mock_engine.enhance.return_value = out_pil
        mock_get_engine.return_value = mock_engine

        result = enhance_image(file_bytes=img_bytes, scale=scale, filename="test.png")

        assert result["success"] is True
        assert result["scale"] == scale
        assert result["input_size"] == (in_w, in_h)
        assert result["output_size"] == (out_w, out_h)
        assert isinstance(result["output_bytes"], bytes)
        assert len(result["output_bytes"]) > 0
        assert result["inference_time"] >= 0.0

    def test_enhancement_service_invalid_scale(self):
        img_bytes = create_dummy_image_bytes(32, 32, format="PNG")
        with pytest.raises(ValueError) as exc_info:
            enhance_image(file_bytes=img_bytes, scale=5)
        assert "Unsupported scale" in str(exc_info.value)
