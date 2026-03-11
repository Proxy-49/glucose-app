# app_glucose_only.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --------------------------
# Streamlit UI config
# --------------------------
st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")
st.title("Glucose Estimator from Microfluidic Bubbles")

# --------------------------
# RGB to HSV conversion
# --------------------------
def rgb_to_hsv(rgb):
    """Convert Nx3 array of RGB [0-1] to HSV [0-1]"""
    rgb = np.array(rgb)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    s[maxc == 0] = 0
    rc = (maxc - rgb[:,0]) / (maxc - minc + 1e-6)
    gc = (maxc - rgb[:,1]) / (maxc - minc + 1e-6)
    bc = (maxc - rgb[:,2]) / (maxc - minc + 1e-6)
    h = np.zeros_like(maxc)
    mask = maxc == rgb[:,0]
    h[mask] = (bc - gc)[mask]
    mask = maxc == rgb[:,1]
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = maxc == rgb[:,2]
    h[mask] = 4.0 + (gc - rc)[mask]
    h = (h / 6.0) % 1.0
    h[minc == maxc] = 0.0
    return np.stack([h, s, v], axis=1)

# --------------------------
# Bubble feature extraction (no visualization)
# --------------------------
def extract_bubble_features(image_path, top_n=20):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=35, minRadius=5, maxRadius=50
    )
    if circles is None:
        raise ValueError("No bubbles detected.")

    circles = np.around(circles).astype(int)
    candidates = []

    for x, y, r in circles[0]:
        r = int(r * 0.9)
        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)
        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if (h_mean >= 252/360) and (h_mean <= 290/360) and (s_mean >= 0.07) and (v_mean >= 0.60):
            score = (h_mean**8)* s_mean * v_mean  * r
            candidates.append({
                "roi_hsv": roi_hsv,
                "score": score
            })

    if len(candidates) == 0:
        raise ValueError("No bubbles passed HSV filter.")

    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)[:top_n]
    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in candidates], axis=0)
    return avg_hsv, img_rgb

# --------------------------
# Calibration data (pure glucose)
# --------------------------
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],
    "H": [0.722795, 0.731712, 0.730700, 0.743624, 0.786134],
    "S": [0.086949, 0.093759, 0.097361, 0.107223, 0.121588]
})

y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H","S"]], y_glucose)

# --------------------------
# Streamlit file uploader
# --------------------------
uploaded_file = st.file_uploader("Upload an image of your microfluidic bubbles", type=["jpg","png","jpeg"])

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_{timestamp}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        avg_hsv, img_rgb = extract_bubble_features(temp_path)
        H_avg, S_avg, V_avg = avg_hsv

        # Convert to DataFrames for prediction
        df_H  = pd.DataFrame({"H":[H_avg]})
        df_S  = pd.DataFrame({"S":[S_avg]})
        df_HS = pd.DataFrame({"H":[H_avg], "S":[S_avg]})

        # Predict glucose
        g_H  = max(model_H.predict(df_H)[0], 0)
        g_S  = max(model_S.predict(df_S)[0], 0)
        g_HS = max(model_HS.predict(df_HS)[0], 0)

        # Weighted glucose using H, S, and H+S model
        glucose_weighted = 0.5*g_H + 0.3*g_S + 0.2*g_HS

        # Display original image only
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        st.subheader("Estimated Glucose (µM)")
        st.write(f"{glucose_weighted:.1f} µM")

    except Exception as e:
        st.error(f"Error processing image: {e}")


