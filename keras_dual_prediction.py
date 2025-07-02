"""predict_dual_input_debug.py - Predict using trained dual-input model with logs + visual debug"""

# Install dependencies
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def preprocess_rgb(image_path, img_size=(128, 128)):
    logging.info("📷 Preprocessing RGB image")
    img = Image.open(image_path).resize(img_size).convert("RGB")
    rgb_array = np.array(img) / 255.0
    rgb_array = np.expand_dims(rgb_array, axis=0)  # Shape: (1, 128, 128, 3)

    # Show RGB image
    plt.imshow(img)
    plt.title("RGB Input")
    plt.axis("off")
    plt.show()

    return rgb_array

def preprocess_fft(image_path, img_size=(128, 128)):
    logging.info("🌐 Computing FFT frequency spectrum")
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, img_size)
    norm = gray / 255.0

    fft = np.fft.fft2(norm)
    fft_shifted = np.fft.fftshift(fft)
    mag_spectrum = np.abs(fft_shifted)
    log_spectrum = 20 * np.log(mag_spectrum + 1e-10)

    # Show the FFT image
    plt.imshow(np.log1p(log_spectrum), cmap='inferno')
    plt.title("FFT Magnitude Spectrum (Log Scaled)")
    plt.axis("off")
    plt.show()

    log_spectrum = log_spectrum.reshape((1, *img_size, 1))  # Shape: (1, 128, 128, 1)
    return log_spectrum

def predict(model_path, image_path):
    logging.info(f"🔍 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    logging.info(f"🖼️ Loading image: {image_path}")
    rgb = preprocess_rgb(image_path)
    fft = preprocess_fft(image_path)

    logging.info("🧠 Making prediction ...")
    prediction = model.predict([rgb, fft])[0][0]
    label = "FAKE" if prediction > 0.7 else "REAL"
    confidence = prediction if prediction > 0.7 else 1 - prediction

    logging.info(f"✅ Prediction: {label} ({confidence * 100:.2f}%)")
    return label, confidence

predict("dual_input_model.h5", "test.png")
