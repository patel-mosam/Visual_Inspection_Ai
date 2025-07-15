# 🛠️ Visual Inspection for QC and Assurance

An AI-powered quality control system using deep learning for **automated visual inspection** in manufacturing.  
Built for Veesure Animal Health, Ahmedabad, as part of the Intel AI for Manufacturing Certificate Course.

---

## 📌 Overview

Manual visual inspections are often inconsistent and error-prone.  
This project introduces an anomaly detection system using **Autoencoders**, **SSIM + MSE hybrid metrics**, and a **Streamlit UI**, trained on category-wise MVTec datasets.

---

## 🚀 Features

- 🧠 Category-specific Autoencoder training
- 📈 SSIM + MSE hybrid thresholding for accurate defect detection
- 🧪 Accuracy evaluation and auto-threshold tuning
- 📸 Streamlit UI with defect preview
- 🔄 Batch training for all MVTec categories
- 🧰 Modular codebase (easy to extend or adapt)

---

## 🗂️ Project Structure

visual-inspection-qc/
├── app.py # Streamlit-based UI
├── autoencoder.py # Model architecture (Conv Autoencoder)
├── train.py # Train one category
├── train_all.py # Batch training for all categories
├── predict.py # Predict defect status
├── evaluate_thresholds.py # Evaluate accuracy, thresholds
├── utils.py # Preprocessing, helpers
├── requirements.txt # Required Python packages
├── README.md # Project documentation


---

## 🖥️ Getting Started

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt


streamlit run app.py

🧪 Training
To train on a specific category:

python train.py --category bottle

- To train all categories & save thresholds:

Copy code
python train_all.py

🧠 Dataset

This project uses the MVTec Anomaly Detection Dataset, which includes:

15 industry-relevant categories

High-quality "good" and "defective" image samples


🧰 Tech Stack

Python 3.9+
TensorFlow / Keras
Streamlit
OpenCV
NumPy
scikit-image (for SSIM calculation)


🤝 Credits
Developed by Mosam Patel
For Veesure Animal Health, Ahmedabad
Under the Intel AI for Manufacturing Certificate Course

🏁 Future Improvements

Replace SSIM + MSE with a learned similarity metric
Add anomaly heatmaps with Grad-CAM
Deploy model backend API using FastAPI or Flask