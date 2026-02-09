import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skimage.color import rgb2hsv
from PIL import Image

st.set_page_config(page_title="Saliva Glucose Estimator", layout="centered")

st.title("📷 Saliva Bubble Glucose Estimator")

st.write("Take or upload a bubble image. The app estimates glucose concentration (µM).")

# =========================
# Calibration data (µM)
# =========================
data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.721974, 0.729639, 0.732042, 0.740652, 0.786352],
    "S": [0.086809, 0.092191, 0.092608, 0.101584, 0.112602]
})

# =========================
# Baseline saliva (blank)
# =========================
H_blank_deg = 215
S_blank_percent = 8.2

H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

data["H_corr"] = data["H"] - H_blank
data["S_corr"] = data["S"] - S_blank

y = data["Glucose"].values

model_H = LinearRegression().fit(data[["H_corr"]], y)
model_S = LinearRegression().fit(data[["S_corr"]], y)
model_HS = LinearRegression().fit(data[["H_corr", "S_corr"]], y)

# =========================
# Image upload
# =========================
uploaded_file = st.file_uploader(
    "📸 Take or upload a saliva bubble image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    hsv_img = rgb2hsv(img_np / 255.0)

    h_mean = hsv_img[..., 0].mean()
    s_mean = hsv_img[..., 1].mean()

    H_corr_input = h_mean - H_blank
    S_corr_input = s_mean - S_blank

    df_H = pd.DataFrame({"H_corr": [H_corr_input]})
    df_S = pd.DataFrame({"S_corr": [S_corr_input]})
    df_HS = pd.DataFrame({"H_corr": [H_corr_input], "S_corr": [S_corr_input]})

    glucose_H = model_H.predict(df_H)[0]
    glucose_S = model_S.predict(df_S)[0]
    glucose_HS = model_HS.predict(df_HS)[0]

    st.subheader("🧪 Estimated Glucose (µM)")
    st.write(f"**Hue only:** {glucose_H:.1f}")
    st.write(f"**Saturation only:** {glucose_S:.1f}")
    st.write(f"**Hue + Saturation:** {max(glucose_HS, 0):.1f}")
