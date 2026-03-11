# Dataset Setup Guide

## Option 1: Automatic Download (Kaggle API)

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
python data/download_data.py --disease all
```

## Option 2: Manual Setup

### TB Dataset
1. Download from: https://www.kaggle.com/datasets/usmanshams/tbx-11
2. Organize as:
```
data/tb/train/tb/        ← TB positive chest X-rays
data/tb/train/normal/    ← Normal chest X-rays
data/tb/val/tb/
data/tb/val/normal/
data/tb/test/tb/
data/tb/test/normal/
```

### MS Dataset
1. Download from: https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis
2. Organize as:
```
data/ms/train/ms/        ← MS positive brain MRIs
data/ms/train/normal/    ← Normal brain MRIs
data/ms/val/ms/
data/ms/val/normal/
data/ms/test/ms/
data/ms/test/normal/
```

## Recommended Split: 70% train / 15% val / 15% test
