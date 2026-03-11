"""
train.py
========
Full two-phase training pipeline for TB and MS classification.

Usage:
    python src/train.py --disease tb --epochs 30 --batch_size 32
    python src/train.py --disease ms --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, RESULTS_DIR, DISEASE_CONFIG, BATCH_SIZE,
    EPOCHS_PHASE1, EPOCHS_PHASE2,
    LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2,
    PATIENCE_EARLY_STOP, PATIENCE_REDUCE_LR, MIN_LR
)
from src.preprocessing import build_data_generators, get_class_weights
from src.model import build_model, unfreeze_for_finetuning


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

def build_callbacks(disease: str, phase: int) -> list:
    """
    Create training callbacks for a given phase.

    Args:
        disease: 'tb' or 'ms'
        phase: 1 or 2

    Returns:
        List of Keras callbacks
    """
    cfg = DISEASE_CONFIG[disease]
    checkpoint_path = cfg["model_path"].replace(".h5", f"_phase{phase}_best.h5")

    callbacks = [
        # Save only the best model weights
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if no improvement
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, f"{disease}_phase{phase}_logs"),
            histogram_freq=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────

def train_phase1(model, train_gen, val_gen, class_weights: dict,
                  epochs: int = EPOCHS_PHASE1) -> tf.keras.callbacks.History:
    """Phase 1: Train only custom head layers with frozen base."""
    print("\n" + "="*60)
    print("🚀 PHASE 1: Training Custom Head (Base Frozen)")
    print("="*60)
    print(f"   Epochs      : {epochs}")
    print(f"   LR          : {LEARNING_RATE_PHASE1}")
    print(f"   Class Weights: {class_weights}")

    # Ensure base is frozen
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # It's a sub-model (EfficientNetB0)
            layer.trainable = False

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=None,  # No checkpoint in phase 1 demo; add if needed
        verbose=1,
    )
    return history


def train_phase2(model, train_gen, val_gen, class_weights: dict,
                  disease: str, epochs: int = EPOCHS_PHASE2) -> tf.keras.callbacks.History:
    """Phase 2: Fine-tune last N layers of EfficientNetB0."""
    print("\n" + "="*60)
    print("🔬 PHASE 2: Fine-Tuning (Partial Unfreeze)")
    print("="*60)
    print(f"   Epochs : {epochs}")
    print(f"   LR     : {LEARNING_RATE_PHASE2}")

    model = unfreeze_for_finetuning(model)
    callbacks = build_callbacks(disease, phase=2)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_final_model(model, disease: str):
    """Save the final trained model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = DISEASE_CONFIG[disease]["model_path"]
    model.save(path)
    print(f"\n💾 Model saved to: {path}")
    return path


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_training_history(history1, history2, disease: str):
    """
    Plot combined training curves for both phases.

    Args:
        history1: Phase 1 training history
        history2: Phase 2 training history
        disease: 'tb' or 'ms'
    """
    cfg = DISEASE_CONFIG[disease]

    # Combine histories
    def combine(h1, h2, key):
        v1 = h1.history.get(key, [])
        v2 = h2.history.get(key, [])
        return v1 + v2

    epochs_total = (
        list(range(1, len(history1.history["loss"]) + 1)) +
        list(range(len(history1.history["loss"]) + 1,
                   len(history1.history["loss"]) + len(history2.history["loss"]) + 1))
    )
    phase1_end = len(history1.history["loss"])

    metrics = [
        ("loss", "val_loss", "Loss", "lower is better"),
        ("accuracy", "val_accuracy", "Accuracy", "higher is better"),
        ("auc", "val_auc", "AUC-ROC", "higher is better"),
        ("precision", "val_precision", "Precision", "higher is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{cfg['name']} — Training History", fontsize=15, fontweight="bold")

    colors = {"train": "#3498db", "val": "#e74c3c"}

    for ax, (train_key, val_key, title, subtitle) in zip(axes.flat, metrics):
        train_vals = combine(history1, history2, train_key)
        val_vals = combine(history1, history2, val_key)

        ax.plot(epochs_total, train_vals, color=colors["train"], linewidth=2,
                label="Train", marker="o", markersize=3)
        ax.plot(epochs_total, val_vals, color=colors["val"], linewidth=2,
                label="Validation", marker="s", markersize=3)

        # Shade phase regions
        ax.axvspan(1, phase1_end, alpha=0.08, color="blue", label="Phase 1 (Frozen)")
        ax.axvspan(phase1_end, epochs_total[-1], alpha=0.08, color="green", label="Phase 2 (Fine-tune)")
        ax.axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.6)

        ax.set_title(f"{title}\n({subtitle})", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{disease}_training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")
    plt.show()


def save_training_log(history1, history2, disease: str):
    """Save training metrics to JSON."""
    log = {
        "disease": disease,
        "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()},
        "final_val_accuracy": float(history2.history["val_accuracy"][-1]),
        "final_val_auc": float(history2.history["val_auc"][-1]),
        "timestamp": datetime.now().isoformat()
    }
    log_path = os.path.join(RESULTS_DIR, f"{disease}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  : {log_path}")


# ─────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────

def train(disease: str, batch_size: int = BATCH_SIZE,
          epochs_p1: int = EPOCHS_PHASE1, epochs_p2: int = EPOCHS_PHASE2):
    """
    Full training pipeline for a given disease model.

    Args:
        disease: 'tb' or 'ms'
        batch_size: Batch size
        epochs_p1: Epochs for Phase 1
        epochs_p2: Epochs for Phase 2
    """
    cfg = DISEASE_CONFIG[disease]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "🔬" * 30)
    print(f"  TRAINING: {cfg['name']}")
    print(f"  Scan Type: {cfg['scan_type']}")
    print("🔬" * 30)

    start_time = time.time()

    # ── 1. Build Data Generators ──────────────────────────
    train_gen, val_gen, test_gen = build_data_generators(disease, batch_size)
    class_weights = get_class_weights(train_gen)
    print(f"\n⚖️  Class Weights: {class_weights}")

    # ── 2. Build Model ────────────────────────────────────
    model = build_model(learning_rate=LEARNING_RATE_PHASE1, trainable_base=False)
    print("\n📐 Model Architecture:")
    model.summary()

    # ── 3. Phase 1: Train Custom Head ────────────────────
    history1 = train_phase1(model, train_gen, val_gen, class_weights, epochs=epochs_p1)

    print(f"\n📊 Phase 1 Results:")
    print(f"   Best Val Accuracy : {max(history1.history['val_accuracy']):.4f}")
    print(f"   Best Val AUC      : {max(history1.history['val_auc']):.4f}")

    # ── 4. Phase 2: Fine-Tune ─────────────────────────────
    history2 = train_phase2(model, train_gen, val_gen, class_weights,
                             disease=disease, epochs=epochs_p2)

    print(f"\n📊 Phase 2 Results:")
    print(f"   Best Val Accuracy : {max(history2.history['val_accuracy']):.4f}")
    print(f"   Best Val AUC      : {max(history2.history['val_auc']):.4f}")

    # ── 5. Save Model ─────────────────────────────────────
    save_final_model(model, disease)

    # ── 6. Save Plots & Logs ──────────────────────────────
    plot_training_history(history1, history2, disease)
    save_training_log(history1, history2, disease)

    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed / 60:.1f} minutes!")
    print(f"   Run evaluation: python src/evaluate.py --disease {disease}")

    return model, history1, history2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TB or MS classification model")
    parser.add_argument("--disease", type=str, required=True, choices=["tb", "ms"],
                        help="Which disease model to train")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Total epochs (split equally between phase 1 and 2)")
    parser.add_argument("--epochs_p1", type=int, default=EPOCHS_PHASE1)
    parser.add_argument("--epochs_p2", type=int, default=EPOCHS_PHASE2)
    args = parser.parse_args()

    if args.epochs:
        p1 = args.epochs // 3
        p2 = args.epochs - p1
    else:
        p1, p2 = args.epochs_p1, args.epochs_p2

    # Set GPU memory growth to avoid OOM
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPUs available: {len(gpus)}")
    else:
        print("⚠️  No GPU found. Training on CPU (will be slow).")

    train(disease=args.disease, batch_size=args.batch_size, epochs_p1=p1, epochs_p2=p2)
