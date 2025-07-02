"""fourier_prediction.py - main script for detecting GAN artifacts and deep fakes."""

# Install dependencies
import cv2 # install with pip install opencv-python
import logging
import numpy as np
import matplotlib.pyplot as plt
import time

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def detect_frequency_artifacts(image_path):
    """Function to identify GAM artifacts with Fourier Transform"""
    # Logger
    start_time = time.time()
    logging.info("⏳ Starting frequency artifact detection.")
    logging.info(f"⏳ Loading image from: {image_path}")

    # Load and convert images to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.errror("❌ Failed to load image. Please check the file path.")
        raise ValueError("Image could not be loaded.")
    logging.info("✅ Image successfully loaded and converted to grayscale.")
    
    # Apply Fourier Transform
    logging.info("🛠️ Applying the Fourier Transform")
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    logging.info("🎉 Fourier Transform Complete.")
    logging.info("⏳ Generating plots ... please hold... ")

    # Display the original and spectrum
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title ('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Frequency Spectrum')
    
    plt.tight_layout()
    plt.show()
    logging.info("🎉 Plots rendered successfully.")

    # Detect potential GAN patterns
    # High frequency grid = suspicious
    spectrum_mean = np.mean(magnitude_spectrum)
    spectrum_std = np.std(magnitude_spectrum)

    logging.info(f"⬜ Mean frequency value: {spectrum_mean:.2f}")
    logging.info(f"⬜ Standard deviation: {spectrum_std:.2f}")

    # Simple heuristic (tunable)
    if spectrum_std > 35 and spectrum_mean > 100:
        logging.warning("⚠️ Suspicious freqneucy patterns detected - possible GAN artifact.")
    else:
        logging.info("✅ No strong GAN-like frequency artifacts detected.")
    
    elapsed_time = time.time() - start_time
    logging.info(f"⬜ Time to run analysis {elapsed_time:.2f} seconds.")
    
# Try it
detect_frequency_artifacts("testimage.png")