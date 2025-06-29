# === app.py ===

import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Dyslexia Detection", layout="centered")
st.title("🧠 Dyslexia Detection from Handwritten Letters")
st.markdown("Upload a clear image of a **handwritten letter** (e.g., capital K). The AI will classify it as **Dyslexic** or **Non-Dyslexic**.")
st.divider()

MODEL_URL = "https://drive.google.com/uc?id=1ttKxvMTEYY8905oe5Yhug5wTa8eptI0X"
MODEL_PATH = "final_dyslexia_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

uploaded_file = st.file_uploader("✏️ Upload Handwritten Letter Image", type=["jpg", "jpeg", "png"])

def predict_dyslexia(img):
    img = img.resize((128, 128)).convert("RGB")
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "🧠 Dyslexic (1)" if prediction > 0.5 else "✅ Non-Dyslexic (0)"
    return label, prediction

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Classifying..."):
        label, confidence = predict_dyslexia(img)

    st.success(f"**{label}**  \nConfidence: `{confidence:.2f}`")
    st.progress(min(confidence, 1.0))
