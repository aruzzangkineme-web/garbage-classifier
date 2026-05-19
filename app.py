import os
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Garbage Classifier",
    page_icon="♻️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .main { background-color: #0f1117; }

    h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -1px; }
    h3 { font-weight: 700 !important; }

    .result-box {
        background: linear-gradient(135deg, #1a1f2e, #222840);
        border: 1px solid #3a3f5c;
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 20px;
        text-align: center;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .result-class {
        font-size: 2.2rem;
        font-weight: 800;
        color: #64ffda;
        text-transform: capitalize;
    }
    .result-conf {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        color: #ccd6f6;
        margin-top: 6px;
    }
    .prob-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #ccd6f6;
        text-transform: capitalize;
        margin-bottom: 2px;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #64ffda, #00bcd4);
        border-radius: 4px;
    }
    div[data-testid="stRadio"] label {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
    }
    .upload-hint {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #8892b0;
        margin-top: 8px;
    }
    .badge {
        display: inline-block;
        background: #1e2a3a;
        border: 1px solid #3a4a5c;
        border-radius: 20px;
        padding: 3px 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #64ffda;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES   = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE  = 224

CLASS_EMOJI = {
    'cardboard': '📦',
    'glass':     '🍶',
    'metal':     '🥫',
    'paper':     '📄',
    'plastic':   '🧴',
    'trash':     '🗑️',
}

# ── Google Drive file IDs ──────────────────────────────────────────────────────
CUSTOM_CNN_ID   = "1t0oLdi1HDuBjgxYGnRPCrOnW6udDzK0z"
INCEPTIONV3_ID  = "1SDEUoHW-WiWK7cUkJHs1uMDabLChArz6"

CUSTOM_CNN_PATH  = "custom_cnn_garbage.keras"
INCEPTIONV3_PATH = "inceptionv3_garbage.keras"

# ── Download model from Google Drive if not already cached ────────────────────
def download_model(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading {output_path}… (first launch only)"):
            gdown.download(url, output_path, quiet=False)

# ── Model loader (cached — loads only once per session) ───────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(file_id: str, path: str):
    download_model(file_id, path)
    return tf.keras.models.load_model(path)

# ── Preprocessing (must match training: rescale=1/255, size=224) ──────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# ♻️ Garbage Classifier")
st.markdown(
    "Upload a photo of waste and the model will identify its category — "
    "helping sort it for proper recycling or disposal."
)

# Class badges
st.markdown(
    " ".join(f'<span class="badge">{CLASS_EMOJI[c]} {c}</span>' for c in CLASSES),
    unsafe_allow_html=True,
)
st.markdown("---")

# Model selector
model_choice = st.radio(
    "**Choose model**",
    ["Custom CNN", "InceptionV3"],
    horizontal=True,
    help="Custom CNN is trained from scratch. InceptionV3 uses ImageNet transfer learning.",
)

if model_choice == "Custom CNN":
    model = load_model(CUSTOM_CNN_ID, CUSTOM_CNN_PATH)
else:
    model = load_model(INCEPTIONV3_ID, INCEPTIONV3_PATH)

st.success(f"✅ **{model_choice}** loaded — {model.count_params():,} parameters")

# Upload
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supported formats: JPG, JPEG, PNG</p>', unsafe_allow_html=True)

# Prediction
if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        with st.spinner("Classifying…"):
            arr   = preprocess(image)
            preds = model.predict(arr, verbose=0)[0]

        pred_idx   = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx])
        emoji      = CLASS_EMOJI[pred_class]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Detected category</div>
            <div class="result-class">{emoji} {pred_class}</div>
            <div class="result-conf">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("### All class probabilities")
    sorted_idx = np.argsort(preds)[::-1]
    for i in sorted_idx:
        st.markdown(
            f'<p class="prob-label">{CLASS_EMOJI[CLASSES[i]]} {CLASSES[i]}</p>',
            unsafe_allow_html=True,
        )
        st.progress(float(preds[i]), text=f"{preds[i]:.2%}")
