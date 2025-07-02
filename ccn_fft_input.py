"""ccn_fft_input.py - script to train cnn on detecting GAN artifacts and deep fakes."""

# Install dependencies
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

class NpyDataGenerator(Sequence):
    def __init__(
            self,
            data_dir,
            batch_size=32,
            img_size=(128,128),
            split='train',
            val_split=0.2,
            shuffle=True
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.split = split
        self.real_paths = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith(".npy")]
        self.fake_paths = [os.path.join(data_dir, "fake", f) for f in os.listdir(os.path.join(data_dir, "fake")) if f.endswith(".npy")]

        # Combine and split data
        self.data = [(p, 0) for p in self.real_paths] + [(p, 1) for p in self.fake_paths]
        random.shuffle(self.data)
        split_idx = int(len(self.data) * (1 - val_split))
        self.data = self.data[:split_idx] if split == 'train' else self.data[split_idx:]

        self.indicies = np.arange(len(self.data))
    
    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for path, label in batch_data:
            arr = np.load(path)
            arr = arr.reshape((*self.img_size, 1)) # (128, 128, 1)
            x_batch.append(arr)
            y_batch.append(label)
        return np.array(x_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)

logging.info("🚀 Starting CNN-based Deefake Finder Pipeline")

# Data parameters for CNN
img_size = 128
batch_size = 32
epochs = 10
dataset_path = 'image_data'

# Data generators
logging.info("📂 Loading frequency spectrum data from .npy files ...")

fft_dataset_path = "frequency_data"

train_data = NpyDataGenerator(
    fft_dataset_path, 
    split='train', 
    batch_size=batch_size)

val_data = NpyDataGenerator(
    fft_dataset_path, 
    split='val', 
    batch_size=batch_size)

logging.info(f"✅ Loaded {len(train_data)*batch_size} training samples and {len(val_data)*batch_size} validation samples.")

# CNN architecture
logging.info("🧠 Building CNN model ...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
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
checkpoint_path = "deepfake_find_fft.h5"
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
