import streamlit as st
from PIL import Image
import numpy as np
import io
import requests
import base64
import time
import os

from detector import ObjectDetectionModel

# ─────────────────────────────────────────────────────────────────────────────
# Initialize model
# ─────────────────────────────────────────────────────────────────────────────
detector = ObjectDetectionModel()

try:
    SUPPORTED_LABELS = list(detector.model_seragam.names.values())
except AttributeError:
    SUPPORTED_LABELS = list(detector.model_seragam.names)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Seragam Detection", layout="centered")
st.title("Seragam Detection")

# Sidebar: supported labels
with st.sidebar:
    st.header("Supported Labels")
    if SUPPORTED_LABELS:
        for lbl in SUPPORTED_LABELS:
            st.markdown(f"- **{lbl}**")
    else:
        st.write("No labels found.")

# ─────────────────────────────────────────────────────────────────────────────
# Helper function to run detection
# ─────────────────────────────────────────────────────────────────────────────
def run_detection(np_image: np.ndarray):
    start_time = time.time()

    with st.spinner("Running detection..."):
        raw_result = detector.classify_array(np_image)
        cleaned = detector.map_clean_result(raw_result)

    elapsed = time.time() - start_time

    # Detected labels
    detected_labels = sorted(
        {d.get("labels", "n/a") for d in cleaned.get("cv_attribute", [])}
    )

    st.success(f"✅ Detection complete in {elapsed:.2f} seconds")

    st.subheader("Detected Labels")
    if detected_labels:
        st.write(", ".join(detected_labels))
    else:
        st.write("None detected.")

    st.subheader("Detection Result (JSON)")
    st.json(cleaned)

# ─────────────────────────────────────────────────────────────────────────────
# Input selection
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Choose Input Method")

option = st.radio("Select input type", ["Default Image", "Upload Image", "Image URL"])

default_images = {
    "Densus 88": "/home/ronny/object-detection/assets/densus_88.jpeg",
    "Kebaya": "/home/ronny/object-detection/assets/kebaya.jpeg",
    "Paskibra": "/home/ronny/object-detection/assets/paskibra.jpeg"
}

if option == "Default Image":
    choice = st.selectbox("Choose a default image", list(default_images.keys()))
    if choice:
        image_path = default_images[choice]
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            st.image(img, caption=f"Default Image: {choice}", use_column_width=True)
            run_detection(np.array(img))
        else:
            st.error(f"File not found: {image_path}")

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        run_detection(np.array(img))

elif option == "Image URL":
    url = st.text_input("Enter Image URL")
    if url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Image from URL", use_column_width=True)
            run_detection(np.array(img))
        except Exception as e:
            st.error(f"Failed to load image: {e}")

