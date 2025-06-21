import streamlit as st
import cv2
from cv2 import dnn_superres
import numpy as np
import tempfile
import os

# Initialize the model
@st.cache_resource
def load_model(model_path):
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("lapsrn", 8)

    try:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        st.success("Using GPU acceleration (CUDA)")
    except Exception:
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        st.warning("GPU not available. Using CPU.")
    return sr

# Streamlit UI
st.title("📈 LapSRN Super-Resolution App")
st.write("Upload a low-resolution image and upscale it 8× using a deep learning model.")

uploaded_file = st.file_uploader("Upload an image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img_bytes = uploaded_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    st.image(image, caption="🔍 Original Image", use_column_width=True)

    model_path = r"E:\ISRO\LapSRN_x8.pb"  # Adjust as needed
    sr = load_model(model_path)

    # Upsample
    upscaled = sr.upsample(image)
    bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)

    col1, col2 = st.columns(2)
    with col1:
        st.image(upscaled, caption="🚀 LapSRN Upscaled", use_column_width=True)
        st.download_button("Download LapSRN Result", data=cv2.imencode('.png', upscaled)[1].tobytes(), file_name="lapsrn_upscaled.png")
    with col2:
        st.image(bicubic, caption="📏 Bicubic Upscaled", use_column_width=True)
        st.download_button("Download Bicubic", data=cv2.imencode('.png', bicubic)[1].tobytes(), file_name="bicubic_upscaled.png")
