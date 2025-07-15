import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_images(folder, max_images=1000, img_size=128):
    images, labels = [], []
    for label_name, label_val in [('good', 0), ('defective', 1)]:
        subfolder = "train" if label_val == 0 else "test"
        path = os.path.join(folder, subfolder, label_name)
        if not os.path.exists(path): continue
        for file in os.listdir(path):
            if not file.endswith(('.png', '.jpg')): continue
            img_path = os.path.join(path, file)
            img = load_img(img_path, color_mode="grayscale", target_size=(img_size, img_size))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label_val)
            if len(images) >= max_images: break
    return np.array(images), np.array(labels)

def show_samples(X, reconstructed, errors, y, threshold, n=5):
    for i in range(n):
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")

        plt.subplot(1, 3, 3)
        plt.imshow((X[i] - reconstructed[i]).squeeze(), cmap='hot')
        plt.title(f"Error: {'Anomaly' if errors[i] > threshold else 'OK'}")
        plt.show()
