"""
evaluate.py
===========
Comprehensive evaluation of trained TB and MS models.
Generates: Confusion Matrix, ROC Curve, Precision-Recall Curve,
           Classification Report, and per-class metrics.

Usage:
    python src/evaluate.py --disease tb
    python src/evaluate.py --disease ms --model_path models/custom_model.h5
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, matthews_corrcoef
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISEASE_CONFIG, RESULTS_DIR, PREDICTION_THRESHOLD
from src.preprocessing import build_data_generators
from src.model import load_model


def evaluate_model(model, test_gen, disease: str, threshold: float = PREDICTION_THRESHOLD):
    """
    Run full evaluation on the test set.

    Args:
        model: Trained Keras model
        test_gen: Test data generator
        disease: 'tb' or 'ms'
        threshold: Classification threshold

    Returns:
        Dictionary of all metrics
    """
    cfg = DISEASE_CONFIG[disease]
    class_names = [cfg["negative_class"], cfg["positive_class"]]

    print(f"\n🔍 Evaluating {cfg['name']} model on test set...")
    print(f"   Test samples: {test_gen.samples}")
    print(f"   Threshold   : {threshold}")

    # Get predictions
    test_gen.reset()
    y_proba = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_proba >= threshold).astype(int)
    y_true = test_gen.classes

    # ── Core Metrics ─────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(tp / (tp + fp + 1e-7)),
        "recall": float(tp / (tp + fn + 1e-7)),
        "specificity": float(tn / (tn + fp + 1e-7)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    # ROC-AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    metrics["roc_auc"] = float(auc(fpr, tpr))

    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    metrics["avg_precision"] = float(average_precision_score(y_true, y_proba))

    # Print summary
    print(f"\n{'═'*50}")
    print(f"  EVALUATION RESULTS — {cfg['name'].upper()}")
    print(f"{'═'*50}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1 Score    : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec.   : {metrics['avg_precision']:.4f}")
    print(f"  MCC         : {metrics['mcc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}, FP={fp}")
    print(f"    FN={fn}, TN={tn}")
    print(f"{'═'*50}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return metrics, y_true, y_pred, y_proba, fpr, tpr, precision_curve, recall_curve, cm


def plot_confusion_matrix(cm, class_names: list, disease: str):
    """Plot and save a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Annotate with count + percentage
    annot = np.array([
        [f"{count}\n({pct:.1%})" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_norm)
    ])

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Proportion"},
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{DISEASE_CONFIG[disease]['name']}\nConfusion Matrix",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{disease}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc: float, disease: str):
    """Plot and save the ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
            label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(
        f"{DISEASE_CONFIG[disease]['name']}\nROC Curve",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{disease}_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def plot_precision_recall_curve(precision, recall, avg_prec: float, disease: str):
    """Plot and save the Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.step(recall, precision, where="post", color="#3498db", lw=2.5,
            label=f"PR Curve (AP = {avg_prec:.4f})")
    ax.fill_between(recall, precision, step="post", alpha=0.1, color="#3498db")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(
        f"{DISEASE_CONFIG[disease]['name']}\nPrecision-Recall Curve",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{disease}_pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def save_metrics_report(metrics: dict, disease: str):
    """Save evaluation metrics to JSON file."""
    cfg = DISEASE_CONFIG[disease]
    report = {
        "disease": cfg["name"],
        "scan_type": cfg["scan_type"],
        "metrics": metrics
    }
    path = os.path.join(RESULTS_DIR, f"{disease}_evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {path}")


def run_evaluation(disease: str, model_path: str = None):
    """
    Full evaluation pipeline.

    Args:
        disease: 'tb' or 'ms'
        model_path: Optional custom model path
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = DISEASE_CONFIG[disease]

    # Load model
    path = model_path or cfg["model_path"]
    model = load_model(path)

    # Build test generator
    _, _, test_gen = build_data_generators(disease)

    # Run evaluation
    (metrics, y_true, y_pred, y_proba,
     fpr, tpr, precision, recall, cm) = evaluate_model(model, test_gen, disease)

    class_names = [cfg["negative_class"], cfg["positive_class"]]

    # Generate all plots
    print("\n📊 Generating evaluation plots...")
    plot_confusion_matrix(cm, class_names, disease)
    plot_roc_curve(fpr, tpr, metrics["roc_auc"], disease)
    plot_precision_recall_curve(precision, recall, metrics["avg_precision"], disease)

    # Save report
    save_metrics_report(metrics, disease)

    print(f"\n✅ Evaluation complete! All results saved to: {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TB or MS model")
    parser.add_argument("--disease", required=True, choices=["tb", "ms"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=PREDICTION_THRESHOLD)
    args = parser.parse_args()

    run_evaluation(args.disease, args.model_path)
