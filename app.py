# import streamlit as st
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt

# IMG_SIZE = 128
# DEFAULT_THRESHOLD = 0.01  # Fallback threshold

# # ---------------------- Functions ----------------------

# def preprocess_image(uploaded_file):
#     img = load_img(uploaded_file, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = img_to_array(img) / 255.0
#     return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# def calculate_error(original, reconstructed):
#     return np.mean((original - reconstructed) ** 2)

# def visualize(original, reconstructed):
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))

#     original_img = original.squeeze()
#     reconstructed_img = reconstructed.squeeze()
#     diff = np.abs(original_img - reconstructed_img)
#     defect_mask = (diff > 0.2).astype(np.uint8)  # adjustable pixel-level threshold

#     axes[0].imshow(original_img, cmap='gray')
#     axes[0].set_title("Original")

#     axes[1].imshow(reconstructed_img, cmap='gray')
#     axes[1].set_title("Reconstructed")

#     axes[2].imshow(diff, cmap='hot')
#     axes[2].set_title("Difference Map")

#     axes[3].imshow(defect_mask, cmap='gray')
#     axes[3].set_title("Defect Mask")

#     for ax in axes:
#         ax.axis("off")

#     st.pyplot(fig)

# def get_threshold_for_category(category):
#     # Future enhancement: fetch thresholds per category from a config
#     # For now: return manually-tuned values or fallback
#     thresholds = {
#         "capsule": 0.0015,
#         "bottle": 0.002,
#         "carpet": 0.0012,
#         # Add more categories as needed
#     }
#     return thresholds.get(category, DEFAULT_THRESHOLD)

# # ---------------------- UI ----------------------

# st.set_page_config(page_title="Visual QC Inspection", layout="centered")
# st.title("🧠 Visual Inspection for Quality Control")

# # Load models
# model_files = [f for f in os.listdir("models/") if f.startswith("autoencoder_") and f.endswith(".h5")]
# categories = [f.replace("autoencoder_", "").replace(".h5", "") for f in model_files]

# if not categories:
#     st.warning("⚠️ No trained models found in the 'models/' folder.")
#     st.stop()

# category = st.selectbox("Select MVTec category:", categories)
# model_path = f"models/autoencoder_{category}.h5"
# threshold = get_threshold_for_category(category)

# uploaded_file = st.file_uploader("Upload an image for inspection", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     with st.spinner("🔍 Inspecting image..."):
#         try:
#             model = load_model(model_path)
#         except Exception as e:
#             st.error(f"❌ Failed to load model: {e}")
#             st.stop()

#         input_img = preprocess_image(uploaded_file)
#         reconstructed_img = model.predict(input_img)

#         error = calculate_error(input_img, reconstructed_img)

#         st.subheader("🔎 Prediction Result")
#         if error > threshold:
#             st.error(f"🚨 Defective (Reconstruction Error = {error:.4f})")
#         else:
#             st.success(f"✅ Good (Reconstruction Error = {error:.4f})")

#         st.subheader("📊 Visual Inspection")
#         visualize(input_img, reconstructed_img)



import streamlit as st
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim

IMG_SIZE = 128

# ---------------------- Preprocessing ----------------------
def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ---------------------- Error Metrics ----------------------
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def calculate_ssim_error(original, reconstructed):
    original_img = original.squeeze()
    reconstructed_img = reconstructed.squeeze()
    score, _ = ssim(original_img, reconstructed_img, full=True, data_range=1.0)
    return 1 - score  # SSIM error

# ---------------------- Visualization ----------------------
def visualize(original, reconstructed):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    original_img = original.squeeze()
    reconstructed_img = reconstructed.squeeze()
    diff = np.abs(original_img - reconstructed_img)
    defect_mask = (diff > 0.2).astype(np.uint8)

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original")

    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title("Reconstructed")

    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Difference Map")

    axes[3].imshow(defect_mask, cmap='gray')
    axes[3].set_title("Defect Mask")

    for ax in axes:
        ax.axis("off")

    st.pyplot(fig)

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Visual QC Inspection", layout="centered")
st.title("🧠 Visual Inspection for Quality Control")

# Step 1: Load models
model_files = [f for f in os.listdir("models/") if f.startswith("autoencoder_") and f.endswith(".h5")]
categories = [f.replace("autoencoder_", "").replace(".h5", "") for f in model_files]

if not categories:
    st.warning("⚠️ No trained models found in the 'models/' folder.")
    st.stop()

# Step 2: Select category
category = st.selectbox("Select MVTec category:", sorted(categories))
model_path = f"models/autoencoder_{category}.h5"

# Step 3: Load thresholds from file
try:
    with open("outputs/thresholds_auto.json") as f:
        thresholds_data = json.load(f)
except Exception as e:
    st.error(f"❌ Failed to load thresholds: {e}")
    st.stop()

# Step 4: Get thresholds for selected category
threshold_info = thresholds_data.get(category, {"mse": 0.002, "ssim": 0.2})
mse_thresh = threshold_info["mse"]
ssim_thresh = threshold_info["ssim"]

# Step 5: Optional manual override
override = st.checkbox("⚙️ Manually override thresholds")
if override:
    st.markdown("### 🔧 Adjust Thresholds")
    mse_thresh = st.slider("MSE Threshold", 0.0001, 0.02, value=mse_thresh, step=0.0001, format="%.6f")
    ssim_thresh = st.slider("SSIM Threshold", 0.05, 1.0, value=ssim_thresh, step=0.01, format="%.4f")

# Optional: Show current thresholds
with st.expander("ℹ️ Threshold Info", expanded=False):
    st.markdown(f"""
    **Selected Category:** `{category}`  
    - **MSE Threshold:** `{mse_thresh}`  
    - **SSIM Threshold:** `{ssim_thresh}`
    """)

# Step 6: Upload image for prediction
uploaded_file = st.file_uploader("📤 Upload an image for inspection", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("🔍 Inspecting image..."):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

        input_img = preprocess_image(uploaded_file)
        reconstructed_img = model.predict(input_img)

        mse_error = calculate_mse(input_img, reconstructed_img)
        ssim_error = calculate_ssim_error(input_img, reconstructed_img)

        # Step 7: Decision
        st.subheader("🔎 Prediction Result")
        st.write(f"📊 MSE Error: `{mse_error:.6f}` (Threshold: `{mse_thresh}`)")
        st.write(f"📊 SSIM Error: `{ssim_error:.4f}` (Threshold: `{ssim_thresh}`)")

        # ✅ Final decision using AND condition
        if mse_error > mse_thresh and ssim_error > ssim_thresh:
            st.error("🚨 Defective Sample Detected!")
        else:
            st.success("✅ Good Sample")

        # Step 8: Visual output
        st.subheader("📌 Visual Inspection")
        visualize(input_img, reconstructed_img)
