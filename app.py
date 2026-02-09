import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

# === Step 1: Calibration data from known glucose samples ===
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],  # µM
    "H": [0.722, 0.733, 0.740, 0.730, 0.786],  # normalized hue
    "S": [0.087, 0.092, 0.102, 0.092, 0.113]   # normalized saturation
})

# === Step 2: Baseline from blank saliva ===
H_blank_deg = 215
S_blank_percent = 8.2
H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

# Apply baseline correction
calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank

# Fit linear regression models
model_H = LinearRegression().fit(calibration_data[["H_corr"]], calibration_data["Glucose"])
model_S = LinearRegression().fit(calibration_data[["S_corr"]], calibration_data["Glucose"])

# Multivariate model
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], calibration_data["Glucose"])

# === Step 3: Utility function to convert RGB to HSV ===
def rgb_to_hsv(rgb):
    """Convert Nx3 RGB [0-1] to HSV [0-1]"""
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

# === Step 4: Bubble extraction ===
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

    circles = np.uint16(np.around(circles))
    bubble_data = []

    for c in circles[0,:]:
        x, y, r = int(c[0]), int(c[1]), int(c[2])
        r_shrink = int(r * 0.9)

        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r_shrink**2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)

        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # Relaxed HSV filter for muted saliva
        if 252/360 <= h_mean <= 290/360 and s_mean >= 0.04 and v_mean >= 0.60:
            score = (h_mean ** 8) * r_shrink
            bubble_data.append({"roi_hsv": roi_hsv, "score": score})

    if len(bubble_data) == 0:
        raise ValueError("No bubbles passed HSV filter.")

    # Sort by score and pick top N
    bubble_data = sorted(bubble_data, key=lambda b: b["score"], reverse=True)[:top_n]

    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in bubble_data], axis=0)
    return avg_hsv, img_rgb

# === Step 5: Streamlit UI ===
st.title("Saliva Glucose Estimator")

uploaded_file = st.file_uploader("Upload saliva bubble image", type=["jpg","png","jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"temp_{timestamp}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        avg_hsv, img_rgb = extract_bubble_features(temp_path)

        # Display original image
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Extract H & S
        H_avg, S_avg, _ = avg_hsv
        # Baseline-correct
        H_corr_input = H_avg - H_blank
        S_corr_input = max(S_avg - S_blank, 0.0)

        df_H  = pd.DataFrame({"H_corr":[H_corr_input]})
        df_S  = pd.DataFrame({"S_corr":[S_corr_input]})
        df_HS = pd.DataFrame({"H_corr":[H_corr_input], "S_corr":[S_corr_input]})

        # Predict glucose
        g_H  = max(model_H.predict(df_H)[0], 0)
        g_S  = max(model_S.predict(df_S)[0], 0)
        g_HS = max(model_HS.predict(df_HS)[0], 0)

        # Weighted average (0.6 H, 0.4 S)
        glucose_avg = 0.6 * g_H + 0.4 * g_S

        st.subheader("Estimated Glucose (µM, weighted)")
        st.write(f"{glucose_avg:.1f} µM")

    except Exception as e:
        st.error(f"Error processing image: {e}")
