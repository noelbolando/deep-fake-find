"""frequency_spectrum.py - script to apply FFT transform and compute the magnitude spectrum for images"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load in image
image_path = 'image_data/real/real_1.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Normalize pixel values
img = img / 255.0
# Apply Fast Fourier Transform
f_transform = np.fft.fft2(img)
# Shift the zero-frequency component to center of image
f_transform_shifted = np.fft.fftshift(f_transform)
# Compute the magnitude spectrum (strength of each frequency component)
magnitude_spectrum = np.abs(f_transform)

# Optional: Visualize log-scaled spectrum
plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
plt.title('FFT Magnitude Spectrum')
plt.axis('off')
plt.show()
