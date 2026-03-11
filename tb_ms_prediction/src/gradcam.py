"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for visualizing which regions of the image the model focuses on.

Reference: https://arxiv.org/abs/1610.02391
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISEASE_CONFIG, RESULTS_DIR, IMG_SIZE, GRADCAM_LAYER, GRADCAM_ALPHA


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Automatically find the name of the last convolutional layer.

    For EfficientNetB0, this is typically 'top_conv' inside the sub-model.
    """
    # If model wraps EfficientNetB0
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # Search inside sub-model
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return f"{layer.name}/{sub_layer.name}"
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model!")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = GRADCAM_LAYER,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for an image.

    Args:
        img_array: Preprocessed image of shape (1, H, W, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last conv layer
        pred_index: Class index to explain (None = predicted class)

    Returns:
        Heatmap array of shape (H, W), values in [0, 1]
    """
    # Build a model that outputs both the conv layer and the final prediction
    try:
        # For wrapped EfficientNetB0
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is not None:
            last_conv_layer = base_model.get_layer("top_conv")
            # Create a grad model that maps input → (conv output, final output)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[
                    base_model.get_layer("top_conv").output,
                    model.output
                ]
            )
        else:
            raise ValueError("fallback")

    except Exception:
        # Fallback: search by layer name
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # For binary classification
        class_channel = predictions[:, 0]

    # Compute gradients of class w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dimensions (Global Average Pooling of gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to [0, 1]
    heatmap = tf.nn.relu(heatmap)  # Keep only positive influences
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        original_img: Original image (H, W, 3), values in [0, 255]
        heatmap: Normalized heatmap from make_gradcam_heatmap
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap for heatmap

    Returns:
        Superimposed image (H, W, 3)
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose
    img_float = original_img.astype(np.float32)
    heatmap_float = colored_heatmap.astype(np.float32)
    superimposed = img_float * (1 - alpha) + heatmap_float * alpha
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed


def visualize_gradcam(
    image_path: str,
    model: tf.keras.Model,
    disease: str,
    save_path: Optional[str] = None
):
    """
    Full Grad-CAM visualization pipeline for a single image.

    Args:
        image_path: Path to the input image
        model: Trained Keras model
        disease: 'tb' or 'ms'
        save_path: Optional path to save the figure
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    cfg = DISEASE_CONFIG[disease]

    # Load and preprocess
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_preprocessed = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    prob = model.predict(img_preprocessed, verbose=0)[0][0]
    predicted_class = cfg["positive_class"] if prob >= 0.5 else cfg["negative_class"]
    confidence = prob if prob >= 0.5 else 1 - prob

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_preprocessed, model)
    superimposed = overlay_heatmap_on_image(img_array.astype(np.uint8), heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{cfg['name']} — Grad-CAM Analysis\n"
        f"Prediction: {predicted_class}  (Confidence: {confidence:.2%})",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(img_array.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=axes[1], fraction=0.046
    )

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay (Heatmap + Image)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grad-CAM saved: {save_path}")
    plt.show()

    return heatmap, superimposed, prob
