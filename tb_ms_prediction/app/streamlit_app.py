"""
streamlit_app.py
================
Interactive Streamlit web application for TB and MS prediction.

Launch:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISEASE_CONFIG, PREDICTION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TB & MS Medical AI Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1B3A6B;
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .result-positive {
        background: #FEE2E2;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .result-negative {
        background: #DCFCE7;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 0.9rem;
        font-size: 0.88rem;
        margin-top: 1.5rem;
    }
    .metric-box {
        background: #EFF6FF;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached(disease: str):
    """Load model once and cache it in session."""
    try:
        import tensorflow as tf
        cfg = DISEASE_CONFIG[disease]
        model = tf.keras.models.load_model(cfg["model_path"])
        return model, None
    except FileNotFoundError:
        return None, f"Model not found at `{DISEASE_CONFIG[disease]['model_path']}`.\nPlease train the model first:\n```\npython src/train.py --disease {disease}\n```"
    except Exception as e:
        return None, str(e)


def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocess PIL image for model inference."""
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_from_array(model, img_array: np.ndarray, disease: str, threshold: float):
    """Run inference and return structured result."""
    cfg = DISEASE_CONFIG[disease]
    prob = float(model.predict(img_array, verbose=0)[0][0])
    is_positive = prob >= threshold
    predicted_class = cfg["positive_class"] if is_positive else cfg["negative_class"]
    confidence = prob if is_positive else (1.0 - prob)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        conf_label = "High Confidence"
    elif confidence >= 0.65:
        conf_label = "Moderate Confidence"
    else:
        conf_label = "⚠️ Low Confidence — Clinical review strongly advised"

    return {
        "prediction": predicted_class,
        "probability": prob,
        "confidence": confidence,
        "confidence_label": conf_label,
        "is_positive": is_positive,
    }


def create_probability_bar(probability: float, disease: str) -> plt.Figure:
    """Create a nice probability gauge chart."""
    cfg = DISEASE_CONFIG[disease]
    neg = cfg["negative_class"]
    pos = cfg["positive_class"]

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh([""], [1 - probability], color="#16A34A", height=0.5, label=neg)
    ax.barh([""], [probability], left=[1 - probability], color="#DC2626", height=0.5, label=pos)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.axvline(0.5, color="white", linewidth=2, linestyle="--")
    ax.set_title("Prediction Probability", fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/microscope.png", width=70)
    st.title("⚙️ Settings")

    disease = st.selectbox(
        "Select Disease to Screen",
        options=["tb", "ms"],
        format_func=lambda x: "🫁 Tuberculosis (Chest X-Ray)" if x == "tb" else "🧠 Multiple Sclerosis (Brain MRI)"
    )

    st.divider()

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1, max_value=0.9,
        value=PREDICTION_THRESHOLD, step=0.05,
        help="Probability above this value = positive prediction"
    )

    show_gradcam = st.checkbox("🔥 Show Grad-CAM Heatmap", value=False,
                               help="Visualize which regions the model focuses on")

    st.divider()
    st.markdown("### ℹ️ About")
    cfg = DISEASE_CONFIG[disease]
    st.info(
        f"**{cfg['name']}**\n\n"
        f"{cfg['description']}\n\n"
        f"**Scan Type:** {cfg['scan_type']}"
    )

    st.divider()
    st.markdown("**Model Architecture**")
    st.code("EfficientNetB0\n+ GlobalAvgPool\n+ Dense(128) + BN + Dropout\n+ Dense(64) + BN + Dropout\n+ Dense(1, Sigmoid)", language=None)


# ──────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 TB & MS Medical AI Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a medical scan image for AI-assisted screening.<br>'
    'Powered by EfficientNetB0 Transfer Learning</div>',
    unsafe_allow_html=True
)

tab_predict, tab_info, tab_about = st.tabs(["🖼️ Predict", "📊 Model Info", "👤 About"])

# ─── TAB 1: PREDICT ───────────────────────────────────────────
with tab_predict:
    model, error = load_model_cached(disease)

    if error:
        st.error(f"❌ Model not loaded:\n\n{error}")
        st.stop()

    st.success(f"✅ {cfg['name']} model loaded successfully!")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader(f"📤 Upload {cfg['scan_type']}")
        uploaded_file = st.file_uploader(
            f"Choose a {cfg['scan_type']} image",
            type=["jpg", "jpeg", "png", "bmp"],
            help=f"Upload a {cfg['scan_type'].lower()} image for {cfg['name']} screening"
        )

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    with col_result:
        st.subheader("📋 Analysis Result")

        if uploaded_file is None:
            st.info("👆 Please upload an image to begin analysis.")
        else:
            with st.spinner("🔄 Analyzing image..."):
                img_array = preprocess_pil_image(pil_image)
                time.sleep(0.3)  # Brief pause for UX
                result = predict_from_array(model, img_array, disease, threshold)

            # Result display
            result_class = "result-positive" if result["is_positive"] else "result-negative"
            icon = "🔴" if result["is_positive"] else "🟢"
            st.markdown(
                f'<div class="{result_class}">'
                f'<h3 style="margin:0">{icon} {result["prediction"]}</h3>'
                f'<p style="margin:0.3rem 0 0 0">{result["confidence_label"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Metric columns
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Probability", f"{result['probability']:.2%}")
            with m2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            with m3:
                st.metric("Threshold", f"{threshold:.0%}")

            # Probability bar
            prob_fig = create_probability_bar(result["probability"], disease)
            st.pyplot(prob_fig, use_container_width=True)
            plt.close(prob_fig)

            # Grad-CAM
            if show_gradcam:
                with st.spinner("🔥 Generating Grad-CAM heatmap..."):
                    try:
                        from src.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image
                        import cv2
                        heatmap = make_gradcam_heatmap(img_array, model)
                        original_np = np.array(pil_image.resize((224, 224)).convert("RGB"))
                        overlay = overlay_heatmap_on_image(original_np, heatmap)

                        g1, g2, g3 = st.columns(3)
                        with g1:
                            st.image(original_np, caption="Original", use_column_width=True)
                        with g2:
                            heatmap_img = plt.cm.jet(heatmap)[:, :, :3]
                            st.image((heatmap_img * 255).astype(np.uint8), caption="Heatmap", use_column_width=True)
                        with g3:
                            st.image(overlay, caption="Overlay", use_column_width=True)

                        st.caption("🔥 Red regions = areas the model focused on most for this prediction")
                    except Exception as e:
                        st.warning(f"Grad-CAM unavailable: {e}")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. '
        'It is NOT a substitute for professional medical advice, diagnosis, or treatment. '
        'Always consult a qualified healthcare provider for medical decisions. '
        'AI predictions may be incorrect.'
        '</div>',
        unsafe_allow_html=True
    )


# ─── TAB 2: MODEL INFO ────────────────────────────────────────
with tab_info:
    st.subheader("📊 Model Performance (Expected Benchmarks)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🫁 TB Detection Model")
        st.table({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
            "Score": ["~95%", "~94%", "~96%", "~95%", "~0.98"]
        })

    with col2:
        st.markdown("#### 🧠 MS Detection Model")
        st.table({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
            "Score": ["~93%", "~92%", "~94%", "~93%", "~0.97"]
        })

    st.divider()
    st.subheader("🔧 Training Pipeline")
    st.code("""
Phase 1 (10 epochs): Frozen EfficientNetB0 → Train custom head only
Phase 2 (20 epochs): Unfreeze last 20 layers → Fine-tune with LR=1e-5

Optimizer   : Adam
Loss        : Binary Cross-Entropy
Augmentation: Rotation, Flip, Zoom, Brightness
Callbacks   : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    """, language=None)

    st.subheader("📦 Tech Stack")
    cols = st.columns(4)
    techs = [
        ("TensorFlow 2.x", "Deep Learning framework"),
        ("EfficientNetB0", "Base model (ImageNet)"),
        ("Scikit-learn", "Metrics & preprocessing"),
        ("Streamlit", "Web interface"),
    ]
    for col, (name, desc) in zip(cols, techs):
        with col:
            st.markdown(f'<div class="metric-box"><b>{name}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)


# ─── TAB 3: ABOUT ─────────────────────────────────────────────
with tab_about:
    st.subheader("👤 Project Information")
    st.markdown("""
    **TB & MS Prediction System** — A deep learning project for AI-assisted medical screening.

    #### 🎯 Diseases Covered
    | Disease | Scan Type | Algorithm |
    |---------|-----------|-----------|
    | Tuberculosis (TB) | Chest X-Ray | EfficientNetB0 Transfer Learning |
    | Multiple Sclerosis (MS) | Brain MRI | EfficientNetB0 Transfer Learning |

    #### 🔗 Datasets Used
    - **TB**: [TBX11K Dataset — Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11)
    - **MS**: [Brain MRI Dataset — Kaggle](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)

    #### 📚 References
    - EfficientNet: *Tan & Le (2019)*, arXiv:1905.11946
    - Grad-CAM: *Selvaraju et al. (2017)*, arXiv:1610.02391

    #### 👨‍💻 Author
    **Rama Cha** · GitHub: [@Rama-cha123](https://github.com/Rama-cha123)
    """)
