"""
config.py
=========
Central configuration for the TB & MS Prediction project.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TB_DATA_DIR = os.path.join(DATA_DIR, "tb")
MS_DATA_DIR = os.path.join(DATA_DIR, "ms")

TB_MODEL_PATH = os.path.join(MODELS_DIR, "tb_efficientnet_model.h5")
MS_MODEL_PATH = os.path.join(MODELS_DIR, "ms_efficientnet_model.h5")

# ─────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ─────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10      # Frozen base model
EPOCHS_PHASE2 = 20      # Fine-tuning
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5

# ─────────────────────────────────────────────
# DATA SPLIT
# ─────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
BASE_MODEL = "EfficientNetB0"   # Options: EfficientNetB0, ResNet50, VGG16
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
FINE_TUNE_LAYERS = 20           # Number of last layers to unfreeze for fine-tuning

# ─────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────
TB_CLASSES = {0: "Normal", 1: "Tuberculosis (TB)"}
MS_CLASSES = {0: "Normal", 1: "Multiple Sclerosis (MS)"}

DISEASE_CONFIG = {
    "tb": {
        "name": "Tuberculosis (TB)",
        "data_dir": TB_DATA_DIR,
        "model_path": TB_MODEL_PATH,
        "classes": TB_CLASSES,
        "scan_type": "Chest X-Ray",
        "description": "TB is a bacterial infection detected via chest X-rays.",
        "positive_class": "Tuberculosis",
        "negative_class": "Normal",
    },
    "ms": {
        "name": "Multiple Sclerosis (MS)",
        "data_dir": MS_DATA_DIR,
        "model_path": MS_MODEL_PATH,
        "classes": MS_CLASSES,
        "scan_type": "Brain MRI",
        "description": "MS is a neurological disease detected via brain MRI scans.",
        "positive_class": "Multiple Sclerosis",
        "negative_class": "Normal",
    }
}

# ─────────────────────────────────────────────
# EARLY STOPPING & CALLBACKS
# ─────────────────────────────────────────────
PATIENCE_EARLY_STOP = 7
PATIENCE_REDUCE_LR = 4
MIN_LR = 1e-7

# ─────────────────────────────────────────────
# DATA AUGMENTATION SETTINGS
# ─────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "nearest",
}

# ─────────────────────────────────────────────
# PREDICTION THRESHOLD
# ─────────────────────────────────────────────
PREDICTION_THRESHOLD = 0.5     # Above this → positive class
HIGH_CONFIDENCE_THRESHOLD = 0.85

# ─────────────────────────────────────────────
# GRAD-CAM SETTINGS
# ─────────────────────────────────────────────
GRADCAM_LAYER = "top_conv"      # Last conv layer in EfficientNetB0
GRADCAM_ALPHA = 0.4             # Overlay transparency

# ─────────────────────────────────────────────
# KAGGLE DATASET IDENTIFIERS
# ─────────────────────────────────────────────
KAGGLE_TB_DATASET = "usmanshams/tbx-11"
KAGGLE_MS_DATASET = "buraktaci/multiple-sclerosis"
