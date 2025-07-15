# import argparse
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix

# from utils import load_images, show_samples

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--category", required=True, help="MVTec category to evaluate")
#     args = parser.parse_args()

#     IMG_SIZE = 128
#     category = args.category
#     path = f"mvtec_anomaly_detection/{category}"

#     print(f"[INFO] Loading model and images for: {category}")
#     model = load_model(f"models/autoencoder_{category}.h5")
#     X, y = load_images(path, img_size=IMG_SIZE)

#     print("[INFO] Predicting...")
#     reconstructed = model.predict(X)
#     errors = np.mean((X - reconstructed) ** 2, axis=(1, 2, 3))
#     threshold = np.percentile(errors[y == 0], 95)
#     y_pred = (errors > threshold).astype(int)

#     print("\nClassification Report:")
#     print(classification_report(y, y_pred))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y, y_pred))

#     print("[INFO] Showing sample visualizations...")
#     show_samples(X, reconstructed, errors, y, threshold, n=5)


















import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from utils import load_images, show_samples

def plot_roc_curve(y_true, scores, category):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {category}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"outputs/roc_curve_{category}.png")
    plt.close()

def save_misclassified(X, reconstructed, errors, y_true, y_pred, threshold, category):
    os.makedirs(f"outputs/{category}/false_positive", exist_ok=True)
    os.makedirs(f"outputs/{category}/false_negative", exist_ok=True)

    for i in range(len(X)):
        if y_true[i] == 0 and y_pred[i] == 1:
            # False Positive
            plt.imsave(f"outputs/{category}/false_positive/sample_{i}_err_{errors[i]:.4f}.png", X[i].squeeze(), cmap='gray')
        elif y_true[i] == 1 and y_pred[i] == 0:
            # False Negative
            plt.imsave(f"outputs/{category}/false_negative/sample_{i}_err_{errors[i]:.4f}.png", X[i].squeeze(), cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="MVTec category to evaluate")
    args = parser.parse_args()

    IMG_SIZE = 128
    category = args.category
    path = f"mvtec_anomaly_detection/{category}"

    print(f"[INFO] Loading model and images for: {category}")
    model = load_model(f"models/autoencoder_{category}.h5")
    X, y = load_images(path, img_size=IMG_SIZE)

    print("[INFO] Predicting...")
    reconstructed = model.predict(X)
    errors = np.mean((X - reconstructed) ** 2, axis=(1, 2, 3))
    
    threshold = np.percentile(errors[y == 0], 95)
    y_pred = (errors > threshold).astype(int)

    print(f"\n[INFO] Threshold (95th percentile of good): {threshold:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\n[INFO] ROC AUC Score:")
    print(f"AUC: {roc_auc_score(y, errors):.4f}")

    # Save ROC curve
    plot_roc_curve(y, errors, category)

    # Save misclassified images
    save_misclassified(X, reconstructed, errors, y, y_pred, threshold, category)

    print("[INFO] Showing sample reconstructions...")
    show_samples(X, reconstructed, errors, y, threshold, n=5)
