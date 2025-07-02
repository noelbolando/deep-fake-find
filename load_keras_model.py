"""load_keras_model.py - script to check best Keras model."""

# NOTE: best practice to run after you run cnn.py to generate ideal deepfake_find.h5 scenario

# Install dependencies
from tensorflow.keras.models import load_model

model = load_model("cnn_deepfake_finder/deepfake_find.h5")
model.summary()