"""cnn_dual_input.py - training CNN on RGB and FFT inputs."""

# Install dependencies
import logging
import numpy as np
import os
from PIL import Image
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import random_rotation, random_zoom

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Paths
rgb_dir = "image_data"
fft_dir = "frequency_data"
batch_size = 32
epochs = 20

class DualInputGenerator(Sequence):
    def __init__(
            self, 
            rgb_dir, 
            fft_dir, 
            batch_size=32, 
            img_size=(128, 128), 
            split='train', 
            val_split=0.2, 
            shuffle=True
        ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.split = split

        self.rgb_paths = []
        self.fft_paths = []
        self.labels = []

        for label, category in enumerate(["real", "fake"]):
            rgb_files = [f for f in os.listdir(os.path.join(rgb_dir, category)) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for fname in rgb_files:
                base = os.path.splitext(fname)[0]
                rgb_path = os.path.join(rgb_dir, category, fname)
                fft_path = os.path.join(fft_dir, category, base + ".npy")
                if os.path.exists(fft_path):
                    self.rgb_paths.append(rgb_path)
                    self.fft_paths.append(fft_path)
                    self.labels.append(label)

        # Shuffle & split
        combined = list(zip(self.rgb_paths, self.fft_paths, self.labels))
        random.shuffle(combined)
        self.rgb_paths, self.fft_paths, self.labels = zip(*combined)

        split_idx = int(len(self.rgb_paths) * (1 - val_split))
        if split == 'train':
            self.rgb_paths = self.rgb_paths[:split_idx]
            self.fft_paths = self.fft_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.rgb_paths = self.rgb_paths[split_idx:]
            self.fft_paths = self.fft_paths[split_idx:]
            self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.rgb_paths) // self.batch_size

    def __getitem__(self, index):
        batch_rgb, batch_fft, batch_y = [], [], []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            # Load RGB
            try:
                img = Image.open(self.rgb_paths[i]).resize(self.img_size).convert("RGB")
                img_array = np.array(img) / 255.0
                img_array = random_rotation(img_array, 10)
                img_array = random_zoom(img_array, (0.9, 1.1))
                batch_rgb.append(img_array)

                # Load FFT
                fft_array = np.load(self.fft_paths[i])
                fft_array = fft_array.reshape((*self.img_size, 1))  # Ensure (128,128,1)
                batch_fft.append(fft_array)

                batch_y.append(self.labels[i])
            except Exception as e:
                logging.warning(f"⚠️ Skipping corrupted file: {self.rgb_paths[i]} | {e}")

        logging.debug(f"📦 Generated batch {index + 1}/{self.__len__()} | Samples: {len(batch_rgb)}")
        return (np.array(batch_rgb), np.array(batch_fft)), np.array(batch_y)

def build_dual_input_model(img_size=(128, 128)):
    # RGB branch
    rgb_input = Input(shape=(img_size[0], img_size[1], 3), name='rgb_input')
    x1 = Conv2D(32, (3,3), activation='relu')(rgb_input)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(64, (3,3), activation='relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Flatten()(x1)

    # FFT branch
    fft_input = Input(shape=(img_size[0], img_size[1], 1), name='fft_input')
    x2 = Conv2D(32, (3,3), activation='relu')(fft_input)
    x2 = MaxPooling2D(2,2)(x2)
    x2 = Conv2D(64, (3,3), activation='relu')(x2)
    x2 = MaxPooling2D(2,2)(x2)
    x2 = Flatten()(x2)

    # Combine
    combined = concatenate([x1, x2])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[rgb_input, fft_input], outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generators
train_gen = DualInputGenerator(
    rgb_dir, 
    fft_dir, 
    batch_size=batch_size, 
    split='train'
)

val_gen = DualInputGenerator(
    rgb_dir, 
    fft_dir, 
    batch_size=batch_size, 
    split='val'
)

# Model
logging.info("🧠 Building dual-input CNN model ...")
# Add early stop
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model = build_dual_input_model()
model.summary(print_fn=lambda x: logging.info(x))

logging.info("✅ Model compiled and ready to train.")
logging.info(f"📊 Starting training for {epochs} epochs with batch size {batch_size} ...")

# Checkpoint
checkpoint = ModelCheckpoint(
    "dual_input_model.h5", 
    save_best_only=True, 
    monitor='val_accuracy', 
    verbose=1
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

logging.info("🏁 Training complete. Evaluating on validation data ...")

# Plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy - Dual Input Model")
plt.show()
