# import argparse
# import os
# from autoencoder import build_autoencoder
# from utils import load_images

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--category", required=True, help="MVTec category to train (e.g., capsule, bottle)")
#     args = parser.parse_args()

#     IMG_SIZE = 128
#     category = args.category
#     path = f"mvtec_anomaly_detection/{category}"

#     print(f"[INFO] Loading images for category: {category}")
#     X, y = load_images(path, img_size=IMG_SIZE)
#     X_train = X[y == 0]  # Only good images

#     print("[INFO] Building and training model...")
#     model = build_autoencoder(IMG_SIZE)
#     model.fit(X_train, X_train, epochs=30, batch_size=32, validation_split=0.1)

#     os.makedirs("models", exist_ok=True)
#     model.save(f"models/autoencoder_{category}.h5", save_format="h5")
#     print(f"[INFO] Model saved to models/autoencoder_{category}.h5")




import argparse
import os
import numpy as np
import tensorflow as tf
from autoencoder import build_autoencoder
from utils import load_images

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="MVTec category to train (e.g., capsule, bottle)")
    args = parser.parse_args()

    IMG_SIZE = 128
    category = args.category
    path = f"mvtec_anomaly_detection/{category}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Category path not found: {path}")

    print(f"[INFO] Loading images for category: {category}")
    X, y = load_images(path, img_size=IMG_SIZE)
    X_train = X[y == 0]  # Only good images (label = 0)

    print("[INFO] Building and training model...")
    model = build_autoencoder(IMG_SIZE)
    model.fit(
        X_train, X_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model_path = f"models/autoencoder_{category}.h5"
    model.save(model_path)
    print(f"[INFO] ✅ Model saved to {model_path}")
