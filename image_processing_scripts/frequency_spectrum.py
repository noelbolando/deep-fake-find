"""frequency_spectrum.py - script to apply FFT transform and compute the magnitude spectrum for images"""

# Install dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load in image
image_path = 'image_data/fake/fake_0.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Normalize pixel values
img = img / 255.0
# Apply Fast Fourier Transform
f_transform = np.fft.fft2(img)
# Shift the zero-frequency component to center of image
f_transform_shifted = np.fft.fftshift(f_transform)
# Compute the magnitude spectrum (strength of each frequency component)
magnitude_spectrum = np.abs(f_transform_shifted)

# Optional: Visualize log-scaled spectrum
plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
plt.title('FFT Magnitude Spectrum')
plt.axis('off')
plt.show()

# Log scaling for the magnitude spectrum
magnitude_spectrum_log = 20 * np.log(magnitude_spectrum + 1e-10) # small constant added to avoid log(0)
plt.imshow(np.log1p(magnitude_spectrum_log), cmap='gray')
plt.title('Log Scaling for FFT Magnitude Spectrum')
plt.axis('off')
plt.show()

# Expand the dimensions to match Keras Conv2D input
# Expected CNN input dimensions (heigh, width, 1)
frequency_input = np.expand_dims(magnitude_spectrum_log, axis=-1)

# Convert to TensorFlow tensor
frequency_input_tensor = tf.convert_to_tensor(frequency_input, dtype=tf.float32)
