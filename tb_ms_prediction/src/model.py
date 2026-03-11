"""
model.py
========
Defines the EfficientNetB0-based Transfer Learning model for
binary classification (TB detection and MS detection).

Architecture:
    EfficientNetB0 (ImageNet weights, frozen)
    → GlobalAveragePooling2D
    → Dense(128, ReLU) + BatchNorm + Dropout(0.4)
    → Dense(64, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(1, Sigmoid)

Two-phase training:
    Phase 1: Only train the top dense layers
    Phase 2: Unfreeze last N layers of EfficientNetB0 (fine-tuning)
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INPUT_SHAPE, LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
    DROPOUT_RATE_1, DROPOUT_RATE_2, DENSE_UNITS_1, DENSE_UNITS_2,
    FINE_TUNE_LAYERS
)


def _resolve_weights(weights: str | None = None) -> str | None:
    """Resolve EfficientNet weights source from args/env with safe fallback."""
    if weights is not None:
        return weights

    env_value = os.getenv("TB_MS_MODEL_WEIGHTS", "imagenet").strip().lower()
    if env_value in {"", "none", "null"}:
        return None
    return "imagenet"


def build_model(
    learning_rate: float = LEARNING_RATE_PHASE1,
    trainable_base: bool = False,
    weights: str | None = None,
) -> Model:
    """
    Build the EfficientNetB0 transfer learning model.

    Args:
        learning_rate: Initial learning rate
        trainable_base: Whether to allow base model weight updates
        weights: Weights source for EfficientNetB0 ("imagenet" or None).
                 If None, resolves from TB_MS_MODEL_WEIGHTS env var.

    Returns:
        Compiled Keras Model
    """
    selected_weights = _resolve_weights(weights)

    # ─── Base Model: EfficientNetB0 ───
    try:
        base_model = EfficientNetB0(
            weights=selected_weights,
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
    except Exception as exc:
        if selected_weights == "imagenet":
            print(f"⚠️ Unable to load ImageNet weights ({exc}). Falling back to random initialization.")
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=INPUT_SHAPE,
            )
        else:
            raise

    base_model.trainable = trainable_base

    # ─── Custom Classification Head ───
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(DENSE_UNITS_1, name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(DROPOUT_RATE_1, name="dropout_1")(x)

    x = layers.Dense(DENSE_UNITS_2, name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Dropout(DROPOUT_RATE_2, name="dropout_2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TB_MS_EfficientNetB0")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    return model


def unfreeze_for_finetuning(model: Model, n_layers: int = FINE_TUNE_LAYERS,
                             learning_rate: float = LEARNING_RATE_PHASE2) -> Model:
    """
    Unfreeze the last N layers of the base model for fine-tuning.

    Args:
        model: The trained Phase 1 model
        n_layers: Number of EfficientNetB0 layers to unfreeze from the end
        learning_rate: Lower learning rate for fine-tuning

    Returns:
        Re-compiled model with unfrozen layers
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find base model in architecture!")

    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n🔓 Fine-tuning: {trainable_count}/{len(base_model.layers)} layers trainable")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    return model


def print_model_summary(model: Model):
    """Print a clean model summary."""
    model.summary(line_length=90)
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"\n  Total Parameters      : {total_params:,}")
    print(f"  Trainable Parameters  : {trainable_params:,}")
    print(f"  Non-Trainable Params  : {total_params - trainable_params:,}")


def load_model(model_path: str) -> Model:
    """
    Load a saved model from disk.

    Args:
        model_path: Path to the .h5 model file

    Returns:
        Loaded Keras Model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model
