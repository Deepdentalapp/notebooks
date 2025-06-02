import streamlit as st
import torch
import yaml
from pathlib import Path
from PIL import Image
import tempfile
import os
import numpy as np
import cv2

# ==== Page config ====
st.set_page_config(page_title="AffoDent AI Dental Screening", layout="wide")

st.title("🦷 AffoDent AI Dental Lesion Detector")
st.markdown("Upload an intraoral photo. The model will detect lesions, caries, and more.")

# ==== Load Model ====
@st.cache_resource
def load_model():
    from yolov5.models.common import DetectMultiBackend
    device = 'cpu'
    model = DetectMultiBackend('best.pt', device=device)
    return model

model = load_model()

# ==== Upload Image ====
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        image_path = tmpfile.name
        image.save(image_path)

    # ==== Run Inference ====
    st.info("🧠 Analyzing...")
    results = model(image_path, size=640, augment=False, visualize=False)
    pred = results.pred[0]

    # ==== Draw Boxes ====
    img_annotated = img_bgr.copy()
    names = model.names

    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img_annotated, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)
            cv2.putText(img_annotated, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), caption="🦷 Detected Issues", use_column_width=True)

        st.markdown("### 🔍 Findings")
        for *xyxy, conf, cls in pred:
            st.write(f"🔹 **{names[int(cls)]}** — Confidence: `{conf:.2f}`")
    else:
        st.success("✅ No lesions, caries, or ulcers detected.")
