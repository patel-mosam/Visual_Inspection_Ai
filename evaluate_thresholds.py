import os
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------ CONFIG -----------------
IMG_SIZE = 128
DATA_DIR = "mvtec_anomaly_detection"

MODEL_DIR = "models"
OUTPUT_FILE = "outputs/thresholds_auto.json"
BUFFER_MSE = 0.0002
BUFFER_SSIM = 0.02

# -------------------------------------

def preprocess(path):
    img = load_img(path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def mse(original, recon):
    return np.mean((original - recon) ** 2)

def ssim_error(original, recon):
    o = original.squeeze()
    r = recon.squeeze()
    score, _ = ssim(o, r, full=True, data_range=1.0)
    return 1 - score

# Get all categories
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Dataset folder not found: {DATA_DIR}")

categories = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])
results = {}

print(f"📂 Found categories: {categories}")

for category in categories:
    model_path = os.path.join(MODEL_DIR, f"autoencoder_{category}.h5")
    good_dir = os.path.join(DATA_DIR, category, "test", "good")

    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {category}: Model not found.")
        continue

    if not os.path.exists(good_dir):
        print(f"⚠️ Skipping {category}: test/good folder not found.")
        continue

    model = load_model(model_path)
    all_mse = []
    all_ssim = []

    print(f"🔍 Evaluating: {category}")
    for fname in os.listdir(good_dir):
        fpath = os.path.join(good_dir, fname)
        try:
            img = preprocess(fpath)
            recon = model.predict(img)
            all_mse.append(mse(img, recon))
            all_ssim.append(ssim_error(img, recon))
        except Exception as e:
            print(f"❌ Error with {fpath}: {e}")

    if not all_mse:
        print(f"⚠️ No valid images in {category}/test/good")
        continue

    mse_thresh = max(all_mse) + BUFFER_MSE
    ssim_thresh = max(all_ssim) + BUFFER_SSIM

    results[category] = {
        "mse": round(float(mse_thresh), 6),
        "ssim": round(float(ssim_thresh), 4)
    }

# Save JSON
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Thresholds saved to: {OUTPUT_FILE}")
