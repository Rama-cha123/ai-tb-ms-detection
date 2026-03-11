"""
preprocessing.py
================
Image preprocessing, data augmentation, and data generator creation
for the TB and MS classification pipeline.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMG_SIZE, BATCH_SIZE, AUGMENTATION,
    DISEASE_CONFIG, RANDOM_SEED
)


def get_class_weights(generator) -> dict:
    """
    Compute class weights to handle class imbalance.

    Args:
        generator: Keras ImageDataGenerator

    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    labels = generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))


def build_data_generators(
    disease: str,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator]:
    """
    Create train, validation, and test data generators.

    Args:
        disease: 'tb' or 'ms'
        batch_size: Batch size for generators

    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    cfg = DISEASE_CONFIG[disease]
    data_dir = cfg["data_dir"]

    # ── Preprocessing: normalize pixels to [0, 1]
    # ── Training: apply augmentation to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=AUGMENTATION["rotation_range"],
        width_shift_range=AUGMENTATION["width_shift_range"],
        height_shift_range=AUGMENTATION["height_shift_range"],
        shear_range=AUGMENTATION["shear_range"],
        zoom_range=AUGMENTATION["zoom_range"],
        horizontal_flip=AUGMENTATION["horizontal_flip"],
        brightness_range=AUGMENTATION["brightness_range"],
        fill_mode=AUGMENTATION["fill_mode"],
    )

    # ── Validation/Test: only normalize, NO augmentation
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
        color_mode="rgb",
    )

    val_gen = val_test_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        color_mode="rgb",
    )

    test_gen = val_test_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        color_mode="rgb",
    )

    print(f"\n✅ Data generators created for {disease.upper()}")
    print(f"   Train batches : {len(train_gen)} | Samples: {train_gen.samples}")
    print(f"   Val batches   : {len(val_gen)} | Samples: {val_gen.samples}")
    print(f"   Test batches  : {len(test_gen)} | Samples: {test_gen.samples}")
    print(f"   Class indices : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image array of shape (1, 224, 224, 3)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image from bytes (for API/Streamlit use).

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image array
    """
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def plot_sample_images(generator, disease: str, n_per_class: int = 4, save_path: Optional[str] = None):
    """
    Visualize sample images from each class.

    Args:
        generator: Train data generator
        disease: 'tb' or 'ms'
        n_per_class: Number of images to show per class
        save_path: Path to save the figure
    """
    cfg = DISEASE_CONFIG[disease]
    class_indices = {v: k for k, v in generator.class_indices.items()}

    # Get a batch
    images, labels = next(generator)
    
    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 3, 7))
    fig.suptitle(
        f"{cfg['name']} — Sample {cfg['scan_type']} Images",
        fontsize=15, fontweight="bold", y=1.02
    )

    classes = [0, 1]  # normal, positive
    class_names = [cfg["negative_class"], cfg["positive_class"]]
    colors = ["#27ae60", "#e74c3c"]

    for row, (cls, cls_name, color) in enumerate(zip(classes, class_names, colors)):
        cls_images = images[labels == cls]
        axes[row, 0].set_ylabel(cls_name, fontsize=13, color=color, fontweight="bold")

        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(cls_images):
                ax.imshow(cls_images[col])
            else:
                ax.axis("off")
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to: {save_path}")

    plt.show()


def plot_class_distribution(generator, disease: str, save_path: Optional[str] = None):
    """
    Plot the distribution of classes in the dataset.
    """
    cfg = DISEASE_CONFIG[disease]
    classes = [cfg["negative_class"], cfg["positive_class"]]
    counts = np.bincount(generator.classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{cfg['name']} — Class Distribution", fontsize=14, fontweight="bold")

    colors = ["#2ecc71", "#e74c3c"]

    # Bar chart
    bars = ax1.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Number of Images", fontsize=12)
    ax1.set_title("Class Counts")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontweight="bold")

    # Pie chart
    ax2.pie(counts, labels=classes, colors=colors, autopct="%1.1f%%",
            startangle=90, shadow=True)
    ax2.set_title("Class Ratio")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return dict(zip(classes, counts))
