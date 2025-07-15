# import os
# from autoencoder import build_autoencoder
# from utils import load_images
# from tensorflow.keras.models import save_model

# IMG_SIZE = 128
# EPOCHS = 30
# BATCH_SIZE = 32

# # List of MVTec categories you want to train
# CATEGORIES = [
#     "capsule", "bottle", "cable", "carpet", "pill", 
#     "tile", "transistor", "wood", "zipper"
# ]

# for category in CATEGORIES:
#     print(f"\n[INFO] Training category: {category}")

#     # Path to the category folder
#     path = f"mvtec_anomaly_detection/{category}"
#     if not os.path.exists(path):
#         print(f"[WARNING] Skipping {category} - dataset not found at {path}")
#         continue

#     # Load good images only
#     X, y = load_images(path, img_size=IMG_SIZE)
#     X_train = X[y == 0]

#     if len(X_train) == 0:
#         print(f"[WARNING] No 'good' images found for {category}. Skipping.")
#         continue

#     print(f"[INFO] Loaded {len(X_train)} training images for {category}")
    
#     # Build and train autoencoder
#     model = build_autoencoder(IMG_SIZE)
#     model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

#     # Save model
#     os.makedirs("models", exist_ok=True)
#     model_path = f"models/autoencoder_{category}.h5"
#     save_model(model, model_path)
#     print(f"[INFO] Saved model to {model_path}")


import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

IMG_SIZE = 128
DATA_DIR = "mvtec_anomaly_detection"
MODEL_DIR = "models"
OUTPUT_FILE = "outputs/thresholds_auto.json"

def preprocess_image(path):
    img = load_img(path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def calculate_ssim_error(original, reconstructed):
    orig = original.squeeze()
    recon = reconstructed.squeeze()
    score, _ = ssim(orig, recon, full=True, data_range=1.0)
    return 1 - score

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

thresholds = {}

for model_file in os.listdir(MODEL_DIR):
    if not model_file.endswith(".h5"):
        continue

    category = model_file.replace("autoencoder_", "").replace(".h5", "")
    print(f"\n📦 Processing: {category}")

    model_path = os.path.join(MODEL_DIR, model_file)
    model = load_model(model_path)

    good_dir = os.path.join(DATA_DIR, category, "train", "good")
    image_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    mse_errors = []
    ssim_errors = []

    for path in tqdm(image_files):
        img = preprocess_image(path)
        recon = model.predict(img)

        mse = calculate_mse(img, recon)
        ssim_err = calculate_ssim_error(img, recon)

        mse_errors.append(mse)
        ssim_errors.append(ssim_err)

    mse_thresh = np.percentile(mse_errors, 98)
    ssim_thresh = np.percentile(ssim_errors, 98)

    thresholds[category] = {
        "mse": round(float(mse_thresh), 6),
        "ssim": round(float(ssim_thresh), 4)
    }

# Save thresholds
with open(OUTPUT_FILE, "w") as f:
    json.dump(thresholds, f, indent=4)

print("\n✅ Thresholds saved to:", OUTPUT_FILE)
