""""batch_fft_preprocess.py - batch process real and fake images into FFT log magenitude spectra."""

# Install dependencies
import cv2
import logging
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(asctime)s - %(message)s]',
    datefmt='%H:%M:%S'
)

logging.info("Starting batch FFT preprocessing ...")

# Image parameters
IMG_SIZE = (128, 128)
INPUT_DIR = "image_data"
OUTPUT_DIR = "frequency_data"
CATEGORIES = ["real", "fake"]

# Create output directories
for cat in CATEGORIES:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)
    logging.info(f"Ensured directory exists: {os.path.join(OUTPUT_DIR, cat)}")

# Process images
def process_image(image_path):
    try:
        # Load in images
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be read.")
        img = cv2.resize(img, IMG_SIZE)
        # Normalize image pixels
        img = img / 225.0

        # Apply Fast Fourier Transform
        f_transform = np.fft.fft2(img)
        # Shift the zero-frequency component to center of image
        f_transform_shifted = np.fft.fftshift(f_transform)
        # Compute the magnitude spectrum (strength of each frequency component)
        magnitude_spectrum = np.abs(f_transform_shifted)
        # Log scaling for the magnitude spectrum
        magnitude_spectrum_log = 20 * np.log(magnitude_spectrum + 1e-10) # small constant added to avoid log(0)
        # Min-max normalization
        magnitude_spectrum_log = (magnitude_spectrum_log - magnitude_spectrum_log.min()) / (magnitude_spectrum_log.max() - magnitude_spectrum_log.min())
        
        # Expand the dimensions to match Keras Conv2D input
        # Expected CNN input dimensions (heigh, width, 1)
        return np.expand_dims(magnitude_spectrum_log, axis=-1)
    
    except Exception as e:
        logging.warning(f"Error processing {image_path}: {e}")
        return None
    
# Batch the image process
for cat in CATEGORIES:
    input_folder = os.path.join(INPUT_DIR, cat)
    output_folder = os.path.join(OUTPUT_DIR, cat)

    logging.info(f"Processing {cat} images ...")
    for fname in tqdm(os.listdir(input_folder), desc=f"Processing {cat} images ..."):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, fname)
            fft_tensor = process_image(image_path)

            if fft_tensor is not None:
                base_name = os.path.splitext(fname)[0]
                save_path = os.path.join(output_folder, f"{base_name}.npy")
                np.save(save_path, fft_tensor)
                logging.debug(f"Saved FFT tensor: {save_path}")
            else:
                logging.warning(f"Skipped: {fname}")
