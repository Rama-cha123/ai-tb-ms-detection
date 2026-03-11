"""
download_data.py
================
Downloads TB and MS datasets from Kaggle and organises them
into the expected directory structure.

Usage:
    python data/download_data.py --disease tb
    python data/download_data.py --disease ms
    python data/download_data.py --disease all
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, TB_DATA_DIR, MS_DATA_DIR,
    KAGGLE_TB_DATASET, KAGGLE_MS_DATASET,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
)

random.seed(RANDOM_SEED)


def check_kaggle_credentials():
    """Verify Kaggle API credentials are available."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("\n📋 To set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Move the downloaded kaggle.json to ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    print("✅ Kaggle credentials found!")
    return True


def download_dataset(kaggle_dataset: str, target_dir: str, dataset_name: str):
    """Download a Kaggle dataset."""
    try:
        import kaggle
        print(f"\n📥 Downloading {dataset_name} dataset...")
        os.makedirs(target_dir, exist_ok=True)
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=target_dir,
            unzip=True
        )
        print(f"✅ {dataset_name} dataset downloaded to: {target_dir}")
        return True
    except ImportError:
        print("❌ Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def split_dataset(source_dir: str, disease_dir: str, classes: list):
    """
    Split a flat dataset into train/val/test folders.

    Args:
        source_dir: Directory containing class subdirectories
        disease_dir: Output directory (e.g., data/tb/)
        classes: List of class folder names
    """
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(disease_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"⚠️  Class directory not found: {cls_dir}")
            continue

        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        print(f"\n  📂 Class: {cls} ({n} images)")
        for split_name, split_images in splits.items():
            dest = os.path.join(disease_dir, split_name, cls)
            for img in tqdm(split_images, desc=f"    → {split_name}", ncols=70):
                shutil.copy2(os.path.join(cls_dir, img), os.path.join(dest, img))
            print(f"    ✅ {split_name}: {len(split_images)} images")


def setup_tb_dataset():
    """Download and organise the TB dataset."""
    print("\n" + "="*60)
    print("🫁 SETTING UP TUBERCULOSIS DATASET")
    print("="*60)

    raw_dir = os.path.join(DATA_DIR, "tb_raw")
    success = download_dataset(KAGGLE_TB_DATASET, raw_dir, "TB Chest X-Ray")

    if not success:
        print("\n📋 Manual Download Instructions:")
        print("  1. Go to: https://www.kaggle.com/datasets/usmanshams/tbx-11")
        print("  2. Download and extract the dataset")
        print("  3. Place images in:")
        print("     data/tb/train/tb/     (TB positive X-rays)")
        print("     data/tb/train/normal/ (Normal X-rays)")
        return

    # Auto-detect structure and organise
    # TBX11K has a specific structure - adapt as needed
    print("\n📂 Organising TB dataset into train/val/test splits...")
    split_dataset(
        source_dir=raw_dir,
        disease_dir=TB_DATA_DIR,
        classes=["tb", "normal"]
    )
    print("\n✅ TB dataset ready!")


def setup_ms_dataset():
    """Download and organise the MS dataset."""
    print("\n" + "="*60)
    print("🧠 SETTING UP MULTIPLE SCLEROSIS DATASET")
    print("="*60)

    raw_dir = os.path.join(DATA_DIR, "ms_raw")
    success = download_dataset(KAGGLE_MS_DATASET, raw_dir, "MS Brain MRI")

    if not success:
        print("\n📋 Manual Download Instructions:")
        print("  1. Go to: https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis")
        print("  2. Download and extract the dataset")
        print("  3. Place images in:")
        print("     data/ms/train/ms/     (MS positive MRI scans)")
        print("     data/ms/train/normal/ (Normal MRI scans)")
        return

    print("\n📂 Organising MS dataset into train/val/test splits...")
    split_dataset(
        source_dir=raw_dir,
        disease_dir=MS_DATA_DIR,
        classes=["ms", "normal"]
    )
    print("\n✅ MS dataset ready!")


def verify_dataset(disease: str):
    """Verify dataset structure after setup."""
    disease_dir = TB_DATA_DIR if disease == "tb" else MS_DATA_DIR
    cls_name = "tb" if disease == "tb" else "ms"
    print(f"\n🔍 Verifying {disease.upper()} dataset structure...")

    total = 0
    all_ok = True
    for split in ["train", "val", "test"]:
        for cls in [cls_name, "normal"]:
            path = os.path.join(disease_dir, split, cls)
            if os.path.exists(path):
                count = len([
                    f for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                total += count
                print(f"  ✅ {split}/{cls}: {count} images")
            else:
                print(f"  ❌ Missing: {path}")
                all_ok = False

    print(f"\n  Total images: {total}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download TB and MS datasets")
    parser.add_argument(
        "--disease", type=str, default="all",
        choices=["tb", "ms", "all"],
        help="Which dataset to download (default: all)"
    )
    args = parser.parse_args()

    print("🔬 TB & MS Prediction — Dataset Downloader")
    print("=" * 60)

    if not check_kaggle_credentials():
        print("\n⚠️  Proceeding with manual setup instructions only.")

    if args.disease in ("tb", "all"):
        setup_tb_dataset()
        verify_dataset("tb")

    if args.disease in ("ms", "all"):
        setup_ms_dataset()
        verify_dataset("ms")

    print("\n🎉 Dataset setup complete! You can now run training:")
    print("   python src/train.py --disease tb")
    print("   python src/train.py --disease ms")


if __name__ == "__main__":
    main()
