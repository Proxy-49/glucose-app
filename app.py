import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image

st.title("Saliva Glucose Estimator")

# === Calibration data (Glucose in µM) ===
data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H_corr": [0.721974, 0.732042, 0.732042, 0.740652, 0.786352],
    "S_corr": [0.086809, 0.092608, 0.092608, 0.101584, 0.112602]
})

y = data["Glucose"].values

# Fit regression models
model_H = LinearRegression().fit(data[["H_corr"]], y)
model_S = LinearRegression().fit(data[["S_corr"]], y)
model_HS = LinearRegression().fit(data[["H_corr","S_corr"]], y)

# === Baseline from blank saliva ===
H_blank_deg = 215      # measured from blank saliva
S_blank_percent = 8.2
H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

# === Upload saliva image ===
uploaded_file = st.file_uploader("Upload a saliva bubble image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    # Read image as RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert to HSV (OpenCV format)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Extract hue and saturation from pink bubbles
    hue_channel = img_hsv[:,:,0] / 179.0      # Normalize 0–1
    sat_channel = img_hsv[:,:,1] / 255.0      # Normalize 0–1

    # Minimal thresholds to avoid gray/white background
    mask = (hue_channel > 0.7) & (sat_channel > 0.06)
    if np.sum(mask) == 0:
        st.warning("No pink bubbles detected. Try another image.")
    else:
        H_mean = np.mean(hue_channel[mask])
        S_mean = np.mean(sat_channel[mask])

        # Baseline correction
        H_corr_input = max(H_mean - H_blank, 0)
        S_corr_input = max(S_mean - S_blank, 0)

        # Predict glucose using three models
        glucose_H = model_H.predict(pd.DataFrame({"H_corr":[H_corr_input]}))[0]
        glucose_S = model_S.predict(pd.DataFrame({"S_corr":[S_corr_input]}))[0]
        glucose_HS = model_HS.predict(pd.DataFrame({"H_corr":[H_corr_input], "S_corr":[S_corr_input]}))[0]

        # Weighted average (example: H+S more weight than single channels)
        final_glucose = (glucose_H + glucose_S + 2*glucose_HS) / 4

        st.success(f"**Estimated Saliva Glucose:** {final_glucose:.1f} µM")

        # Show uploaded image
        st.image(image, caption="Uploaded saliva bubble image", use_column_width=True)
