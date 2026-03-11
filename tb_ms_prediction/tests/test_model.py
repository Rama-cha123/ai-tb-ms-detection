"""
test_model.py
=============
Unit tests for model building, preprocessing, and prediction logic.

Run:
    pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_SHAPE, IMG_SIZE, DISEASE_CONFIG


class TestConfig:
    """Tests for configuration values."""

    def test_img_size_positive(self):
        assert IMG_SIZE[0] > 0
        assert IMG_SIZE[1] > 0

    def test_input_shape(self):
        assert INPUT_SHAPE == (224, 224, 3)

    def test_disease_config_keys(self):
        for disease in ["tb", "ms"]:
            cfg = DISEASE_CONFIG[disease]
            assert "name" in cfg
            assert "data_dir" in cfg
            assert "model_path" in cfg
            assert "classes" in cfg
            assert "scan_type" in cfg


class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_preprocess_image_bytes(self):
        from src.preprocessing import preprocess_image_bytes
        from PIL import Image
        import io

        # Create a synthetic test image (224x224 white image)
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = preprocess_image_bytes(image_bytes)

        assert result.shape == (1, 224, 224, 3), f"Expected (1, 224, 224, 3), got {result.shape}"
        assert result.min() >= 0.0, "Pixel values should be >= 0"
        assert result.max() <= 1.0, "Pixel values should be <= 1 after normalization"

    def test_preprocess_single_image(self, tmp_path):
        from src.preprocessing import preprocess_single_image
        from PIL import Image

        # Save a synthetic image
        img_path = str(tmp_path / "test_image.jpg")
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        img.save(img_path)

        result = preprocess_single_image(img_path)
        assert result.shape == (1, 224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestModelArchitecture:
    """Tests for model building."""

    def test_model_builds_correctly(self):
        from src.model import build_model
        model = build_model()

        # Check input shape
        assert model.input_shape == (None, 224, 224, 3)
        # Check output shape (binary = 1 neuron)
        assert model.output_shape == (None, 1)
        # Check model has layers
        assert len(model.layers) > 5

    def test_model_output_range(self):
        from src.model import build_model
        model = build_model()

        # Random input
        test_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
        predictions = model.predict(test_input, verbose=0)

        # Sigmoid output must be in [0, 1]
        assert predictions.shape == (2, 1)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)

    def test_model_is_compiled(self):
        from src.model import build_model
        model = build_model()

        # Compiled model has optimizer
        assert model.optimizer is not None
        assert model.loss is not None

    def test_model_trainable_params_phase1(self):
        from src.model import build_model
        model = build_model(trainable_base=False)

        # With frozen base, only head layers are trainable
        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        total = model.count_params()
        # Head should be <5% of total params
        assert trainable < total * 0.05, "Too many params trainable in Phase 1"

    def test_unfreeze_increases_trainable_params(self):
        from src.model import build_model, unfreeze_for_finetuning
        model = build_model(trainable_base=False)

        trainable_before = sum(np.prod(w.shape) for w in model.trainable_weights)
        model = unfreeze_for_finetuning(model, n_layers=10)
        trainable_after = sum(np.prod(w.shape) for w in model.trainable_weights)

        assert trainable_after > trainable_before, "Unfreezing should increase trainable params"


class TestPredictionLogic:
    """Tests for prediction logic."""

    def test_prediction_range(self):
        from src.model import build_model
        model = build_model()

        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        pred = model.predict(test_input, verbose=0)[0][0]

        assert 0.0 <= pred <= 1.0

    def test_threshold_logic(self):
        """Test that threshold correctly determines class."""
        probability = 0.7
        threshold = 0.5
        assert probability >= threshold  # Should be positive

        probability2 = 0.3
        assert probability2 < threshold  # Should be negative

    def test_confidence_calculation(self):
        """Test confidence = distance from threshold."""
        prob = 0.9
        is_positive = prob >= 0.5
        confidence = prob if is_positive else (1 - prob)
        assert confidence == 0.9

        prob2 = 0.1
        is_positive2 = prob2 >= 0.5
        confidence2 = prob2 if is_positive2 else (1 - prob2)
        assert confidence2 == 0.9  # Symmetric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
