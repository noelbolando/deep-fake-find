"""keras_predictions.py - script to make predictions of deepfake identified images."""

# Install dependencies
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model("deepfake_find.h5")
model.summary()

def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    print(f"{label} ({prediction:.2%} confidence)")

predict_image("IMG_1459.JPG")   
