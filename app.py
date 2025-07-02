"""app.py - streamlit-powered app for interfacing with Keras predictor model"""

# Install dependencies
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load Keras Model
@st.cache_resource
def load_cnn_model():
    model = load_model("deepfake_find.h5")
    return model

model = load_cnn_model()

# Prediction Function
def predict_image(image: Image.Image):
    img = image.resize((128, 128)).convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "🟥 Fake" if prediction > 0.5 else "🟩 Real"
    confidence = f"{prediction:.2%}" if prediction > 0.5 else f"{(1 - prediction):.2%}"
    return label, confidence

# Steamlit UI
st.set_page_config(page_title="DeepFake Finder", layout="centered")

st.title("🧠 DeepFake Finder")
st.write("Upload an image to check if it's real or GAN-generated.")

uploaded_file = st.file_uploader("📂 Choose an image ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("⏳ Analyzing...")

    label, confidence = predict_image(image)
    st.subheader(f"🔮 Prediction: {label}")
    st.write(f"📊 Confidence: {confidence}")
