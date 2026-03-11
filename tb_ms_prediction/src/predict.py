"""
predict.py
==========
Run inference on new X-ray / MRI images using trained models.

Usage:
    python src/predict.py --disease tb --image path/to/xray.jpg
    python src/predict.py --disease ms --image path/to/mri.png --gradcam
    python src/predict.py --disease tb --image path/to/xray.jpg --threshold 0.6
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DISEASE_CONFIG, RESULTS_DIR,
    PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    IMG_SIZE
)
from src.model import load_model
from src.preprocessing import preprocess_single_image


def predict_single(
    image_path: str,
    model,
    disease: str,
    threshold: float = PREDICTION_THRESHOLD,
    show_gradcam: bool = False
) -> dict:
    """
    Predict disease from a single image.

    Args:
        image_path: Path to the input image file
        model: Loaded Keras model
        disease: 'tb' or 'ms'
        threshold: Classification threshold (default 0.5)
        show_gradcam: Whether to show Grad-CAM explanation

    Returns:
        Dictionary with prediction results
    """
    cfg = DISEASE_CONFIG[disease]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Preprocess
    img_array = preprocess_single_image(image_path)

    # Predict
    probability = float(model.predict(img_array, verbose=0)[0][0])
    is_positive = probability >= threshold
    predicted_class = cfg["positive_class"] if is_positive else cfg["negative_class"]
    confidence = probability if is_positive else (1.0 - probability)

    # Confidence level label
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_label = "High Confidence"
    elif confidence >= 0.65:
        confidence_label = "Moderate Confidence"
    else:
        confidence_label = "Low Confidence — Consider Clinical Review"

    result = {
        "disease": cfg["name"],
        "scan_type": cfg["scan_type"],
        "image_path": image_path,
        "prediction": predicted_class,
        "probability": probability,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "is_positive": is_positive,
        "threshold": threshold,
    }

    # Print result
    print("\n" + "─" * 55)
    print(f"  🔬 {cfg['name']} PREDICTION RESULT")
    print("─" * 55)
    print(f"  Image       : {os.path.basename(image_path)}")
    print(f"  Prediction  : {'🔴 ' if is_positive else '🟢 '}{predicted_class}")
    print(f"  Probability : {probability:.4f} ({probability*100:.2f}%)")
    print(f"  Confidence  : {confidence:.4f} ({confidence*100:.2f}%) — {confidence_label}")
    print(f"  Threshold   : {threshold}")
    print("─" * 55)

    # Medical disclaimer
    print("\n⚠️  DISCLAIMER: This is an AI-assisted screening tool.")
    print("   Always consult a qualified medical professional for diagnosis.")

    # Visualise prediction
    visualize_prediction(image_path, result)

    # Grad-CAM
    if show_gradcam:
        try:
            from src.gradcam import visualize_gradcam
            save_path = os.path.join(
                RESULTS_DIR,
                f"{disease}_gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            )
            os.makedirs(RESULTS_DIR, exist_ok=True)
            visualize_gradcam(image_path, model, disease, save_path=save_path)
        except Exception as e:
            print(f"⚠️  Grad-CAM failed: {e}")

    return result


def visualize_prediction(image_path: str, result: dict):
    """Show the image with prediction overlay."""
    img = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.axis("off")

    color = "#e74c3c" if result["is_positive"] else "#27ae60"
    label = (
        f"{result['prediction']}\n"
        f"Confidence: {result['confidence']:.2%}\n"
        f"{result['confidence_label']}"
    )
    ax.set_title(
        f"{result['disease']} — {result['scan_type']}\n{label}",
        fontsize=13, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.show()


def batch_predict(image_dir: str, disease: str, model=None, threshold: float = PREDICTION_THRESHOLD):
    """
    Run predictions on all images in a directory.

    Args:
        image_dir: Directory containing images
        disease: 'tb' or 'ms'
        model: Loaded model (loaded from config if None)
        threshold: Classification threshold

    Returns:
        List of prediction result dicts
    """
    if model is None:
        cfg = DISEASE_CONFIG[disease]
        model = load_model(cfg["model_path"])

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        print(f"⚠️  No images found in: {image_dir}")
        return []

    print(f"\n📁 Batch prediction on {len(image_files)} images...")
    results = []
    for fname in image_files:
        path = os.path.join(image_dir, fname)
        result = predict_single(path, model, disease, threshold, show_gradcam=False)
        results.append(result)

    # Summary
    positives = sum(1 for r in results if r["is_positive"])
    negatives = len(results) - positives
    print(f"\n📊 Batch Summary:")
    print(f"   Total   : {len(results)}")
    print(f"   Positive: {positives} ({positives/len(results)*100:.1f}%)")
    print(f"   Negative: {negatives} ({negatives/len(results)*100:.1f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict TB or MS from medical images")
    parser.add_argument("--disease", required=True, choices=["tb", "ms"])
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to batch image folder")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=PREDICTION_THRESHOLD)
    parser.add_argument("--gradcam", action="store_true", help="Show Grad-CAM explanation")
    args = parser.parse_args()

    cfg = DISEASE_CONFIG[args.disease]
    model_path = args.model_path or cfg["model_path"]
    model = load_model(model_path)

    if args.image:
        predict_single(args.image, model, args.disease, args.threshold, args.gradcam)
    elif args.image_dir:
        batch_predict(args.image_dir, args.disease, model, args.threshold)
    else:
        print("❌ Please provide --image or --image_dir")
