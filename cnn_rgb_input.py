"""cnn_rgb_input.py - script to train cnn on detecting GAN artifacts and deep fakes."""

# Install dependencies
import logging
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

logging.info("🚀 Starting CNN-based Deefake Finder Pipeline")

# Data parameters for CNN
img_size = 128
batch_size = 32
epochs = 10
dataset_path = 'image_data'

# Data generators
logging.info("📂 Preparing image data generators ...")
train_gen = ImageDataGenerator(
    rescale=1./225, 
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

logging.info(f"✅ Loaded {train_data.samples} training images and {val_data.samples} validation images.")

# CNN architecture
logging.info("🧠 Building CNN model ...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') #binary: 0=real, 1=fake
])

model.compile(
    optimizer=Adam(1e-4), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

logging.info("✅ Model compiled successfully.")
model.summary(print_fn=lambda x: logging.info(x))

# Save best model
checkpoint_path = "deepfake_find_rgb.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
logging.info("🏗️ Starting model training ...")
history = model.fit(
    train_data,
    validation_data=val_data, 
    epochs=epochs,
    callbacks=[checkpoint]
)

logging.info("✅ Training complete")
logging.info(f"✅ Best model saved to: {checkpoint_path}")

# Evaluate the model
logging.info("🏗️ Evaluating model on validation data ...")
loss, acc = model.evaluate(val_data)
logging.info(f"🧪 Final Validation Accuracy: {acc:.4f}")

# Plot the model to evaluate
logging.info(f"🏗️ Plotting the model accuracy ...")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
