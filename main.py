import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from utils import load_images, show_samples

# ========================== CONFIG ==========================
IMG_SIZE = 128
CATEGORY = "capsule"  # Change this to bottle, screw, etc.
DATASET_PATH = f"./mvtec_anomaly_detection/{CATEGORY}"
EPOCHS = 30
BATCH_SIZE = 32

# ======================= Load Data ===========================
print("[INFO] Loading data...")
X, y = load_images(DATASET_PATH)
X_train = X[y == 0]  # Only good samples for training

# ==================== Build Autoencoder ======================
print("[INFO] Building model...")
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Encoder
x = Conv2D(32, 3, activation='relu', padding='same')(input_img)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)

# Decoder
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.summary()

# ===================== Train Model ===========================
print("[INFO] Training model...")
autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True
)

# ===================== Predict & Evaluate ====================
print("[INFO] Evaluating model...")
reconstructed = autoencoder.predict(X)
errors = np.mean((X - reconstructed) ** 2, axis=(1, 2, 3))
threshold = np.percentile(errors[y == 0], 95)
y_pred = (errors > threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# ===================== Visualize Results =====================
print("[INFO] Showing sample results...")
show_samples(X, reconstructed, errors, y, threshold, n=5)

# ===================== Save Model ============================
model_path = f"./models/autoencoder_{CATEGORY}.h5"
autoencoder.save(model_path)
print(f"[INFO] Model saved to {model_path}")
