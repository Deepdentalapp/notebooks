import streamlit as st
from PIL import Image
import torch
import os
import tempfile
from datetime import datetime

# ==== Streamlit App Config ====
st.set_page_config(page_title="AffoDent Oral Screening", layout="wide")

st.title("🦷 AffoDent AI Dental Lesion Detector")
st.markdown("Upload an intraoral image, and the AI will detect lesions, caries, and more.")

# ==== Load YOLOv5 Model ====
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# ==== Upload Image ====
uploaded_file = st.file_uploader("Upload a dental photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
        img.save(temp_img_file.name)
        temp_path = temp_img_file.name

    # ==== Run Inference ====
    st.info("Analyzing image...")
    results = model(temp_path)

    # ==== Annotated Image ====
    st.image(results.render()[0], caption="Detected Lesions", use_column_width=True)

    # ==== Prediction Table ====
    st.markdown("### 📝 Detected Findings")
    df = results.pandas().xyxy[0]
    if df.empty:
        st.success("✅ No visible dental lesions detected.")
    else:
        for idx, row in df.iterrows():
            st.write(f"🔹 **{row['name']}** detected at confidence {row['confidence']:.2f}")

    # ==== Optional: Save Results ====
    save_results = st.checkbox("Download annotated image")
    if save_results:
        annotated_path = os.path.join(tempfile.gettempdir(), f"annotated_{datetime.now().strftime('%H%M%S')}.jpg")
        Image.fromarray(results.render()[0]).save(annotated_path)
        with open(annotated_path, "rb") as f:
            st.download_button("📥 Download", f, file_name="dental_result.jpg")
