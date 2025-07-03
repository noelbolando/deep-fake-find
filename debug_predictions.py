import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

# Config
IMG_SIZE = (128, 128)
DEBUG_DIR = "debug_results"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def preprocess_rgb(image_path):
    img = Image.open(image_path).resize(IMG_SIZE).convert("RGB")
    rgb_array = np.array(img) / 255.0
    return np.expand_dims(rgb_array, axis=0), img

def preprocess_fft(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, IMG_SIZE)
    norm = gray / 255.0
    fft = np.fft.fft2(norm)
    fft_shifted = np.fft.fftshift(fft)
    mag_spectrum = np.abs(fft_shifted)
    log_spectrum = 20 * np.log(mag_spectrum + 1e-10)
    return log_spectrum.reshape((1, *IMG_SIZE, 1)), log_spectrum

def predict_and_debug(model, image_path, fft_path, index, true_label):
    rgb_input, rgb_img = preprocess_rgb(image_path)
    fft_input, fft_img = preprocess_fft(image_path)  # Same image path used

    prediction = model.predict([rgb_input, fft_input])[0][0]
    predicted_label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Logging
    logging.info(f"[{index}] GT: {true_label} | Pred: {predicted_label} ({confidence:.2f}) | File: {os.path.basename(image_path)}")

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb_img)
    axs[0].set_title(f"RGB\nGT: {true_label} | Pred: {predicted_label} ({confidence:.2f})")
    axs[0].axis("off")

    axs[1].imshow(np.log1p(fft_img), cmap='inferno')
    axs[1].set_title("FFT Spectrum")
    axs[1].axis("off")

    output_path = os.path.join(DEBUG_DIR, f"debug_{index}_{true_label}_{predicted_label}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def debug_batch(model_path, data_dir, max_per_class=10):
    model = tf.keras.models.load_model(model_path)
    logging.info(f"✅ Loaded model from: {model_path}")
    logging.info(f"🔍 Starting batch debug from: {data_dir}")

    image_paths = []
    for label in ["real", "fake"]:
        label_dir = os.path.join(data_dir, label)
        images = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith((".jpg", ".png"))])[:max_per_class]
        for img_path in images:
            image_paths.append((img_path, label))

    for idx, (path, true_label) in enumerate(image_paths):
        predict_and_debug(model, path, None, idx, true_label)

    logging.info(f"✅ Debugging complete. Saved results in {DEBUG_DIR}/")

# Run the debug tool
if __name__ == "__main__":
    debug_batch("dual_input_model.h5", "image_data", max_per_class=10)
