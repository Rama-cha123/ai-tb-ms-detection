# 🫁🧠 TB & Multiple Sclerosis Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A complete end-to-end deep learning system for detecting **Tuberculosis (TB)** from chest X-rays and **Multiple Sclerosis (MS)** from brain MRI scans using Transfer Learning with EfficientNetB0.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Web App](#web-app)
- [API](#api)

---

## 🔍 Overview

This project implements two separate binary classifiers:

| Disease | Input | Model | Task |
|---------|-------|-------|------|
| **Tuberculosis (TB)** | Chest X-Ray Images | EfficientNetB0 (Transfer Learning) | Binary Classification: TB / Normal |
| **Multiple Sclerosis (MS)** | Brain MRI Images | EfficientNetB0 (Transfer Learning) | Binary Classification: MS / Normal |

### Key Features
- ✅ Transfer Learning with EfficientNetB0 (pre-trained on ImageNet)
- ✅ Advanced Data Augmentation to handle class imbalance
- ✅ Grad-CAM visualizations to explain predictions
- ✅ Complete training pipeline with callbacks (EarlyStopping, ReduceLROnPlateau)
- ✅ Streamlit web application for real-time prediction
- ✅ REST API using FastAPI
- ✅ Model evaluation with Confusion Matrix, ROC-AUC, Classification Report
- ✅ Jupyter notebooks for EDA and experimentation

---

## 📁 Project Structure

```
tb_ms_prediction/
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── config.py                     # Central configuration
│
├── data/
│   ├── download_data.py          # Auto-download datasets from Kaggle
│   └── README.md                 # Dataset instructions
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Image loading, augmentation, generators
│   ├── model.py                  # Model architecture (EfficientNetB0)
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation metrics & plots
│   ├── predict.py                # Inference on new images
│   └── gradcam.py                # Grad-CAM explainability
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb   # Full training walkthrough
│
├── app/
│   └── streamlit_app.py          # Interactive web application
│
├── api/
│   └── fastapi_app.py            # REST API for predictions
│
├── tests/
│   └── test_model.py             # Unit tests
│
├── models/                       # Saved model weights (.h5)
│   └── .gitkeep
│
└── results/                      # Plots, metrics, confusion matrices
    └── .gitkeep
```

---

## 📊 Dataset

### Tuberculosis Dataset
- **Source**: [TBX11K Dataset (Kaggle)](https://www.kaggle.com/datasets/usmanshams/tbx-11)
- **Alternative**: [NIH Chest X-Ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Classes**: TB (Positive), Normal (Negative)
- **Format**: JPEG/PNG chest X-ray images

### Multiple Sclerosis Dataset
- **Source**: [Brain MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Alternative**: [MS Brain MRI](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)
- **Classes**: MS (Positive), Normal (Negative)
- **Format**: JPEG/PNG brain MRI scan images

> **Note**: MS is clinically diagnosed via brain MRI, not traditional X-rays. This project correctly uses MRI for MS detection and X-rays for TB detection.

### Dataset Directory Structure
```
data/
├── tb/
│   ├── train/
│   │   ├── tb/           # TB positive chest X-rays
│   │   └── normal/       # Normal chest X-rays
│   ├── val/
│   │   ├── tb/
│   │   └── normal/
│   └── test/
│       ├── tb/
│       └── normal/
│
└── ms/
    ├── train/
    │   ├── ms/           # MS positive brain MRI
    │   └── normal/       # Normal brain MRI
    ├── val/
    │   ├── ms/
    │   └── normal/
    └── test/
        ├── ms/
        └── normal/
```

---

## 🧠 Model Architecture

```
Input Image (224x224x3)
        ↓
EfficientNetB0 (Pre-trained on ImageNet, frozen initially)
        ↓
Global Average Pooling 2D
        ↓
Dense (128, ReLU) + BatchNormalization + Dropout(0.4)
        ↓
Dense (64, ReLU) + BatchNormalization + Dropout(0.3)
        ↓
Dense (1, Sigmoid)
        ↓
Output: Probability [0=Normal, 1=Disease]
```

### Training Strategy
1. **Phase 1**: Freeze EfficientNetB0, train only top layers (10 epochs)
2. **Phase 2**: Unfreeze last 20 layers, fine-tune with lower LR (20 epochs)

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| TB Detection | ~95% | ~94% | ~96% | ~95% | ~0.98 |
| MS Detection | ~93% | ~92% | ~94% | ~93% | ~0.97 |

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/Rama-cha123/tb-ms-prediction.git
cd tb-ms-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets
```bash
# Install Kaggle API first
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
python data/download_data.py
```

---

## 🚀 Usage

### Train TB Model
```bash
python src/train.py --disease tb --epochs 30 --batch_size 32
```

### Train MS Model
```bash
python src/train.py --disease ms --epochs 30 --batch_size 32
```

### Evaluate a Model
```bash
python src/evaluate.py --disease tb --model_path models/tb_model.h5
```

### Predict on a Single Image
```bash
python src/predict.py --disease tb --image_path path/to/xray.jpg
```

---

## 🌐 Web App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🔌 API

```bash
uvicorn api.fastapi_app:app --reload
```

Open http://localhost:8000/docs for the Swagger UI.

**Endpoint:**
```
POST /predict
Body: { "disease": "tb", "image": <base64_encoded_image> }
```

---

## 📓 Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Dataset exploration, class distribution, sample visualization |
| `02_Model_Training.ipynb` | Complete training walkthrough with plots |

---

## ☁️ GitHub Hosting & Automation

- GitHub Actions workflow (`../.github/workflows/ci.yml`) automatically runs tests for every push and pull request.
- CI uses `TB_MS_MODEL_WEIGHTS=none` so tests work in network-restricted environments.
- Keep model artifacts out of Git and only track placeholders (`models/.gitkeep`, `results/.gitkeep`).

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Rama Cha**  
GitHub: [@Rama-cha123](https://github.com/Rama-cha123)

---

## 🙏 Acknowledgements

- [TensorFlow / Keras](https://tensorflow.org)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- Kaggle datasets for TB and MS imaging
