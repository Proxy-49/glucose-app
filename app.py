# app.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Saliva Glucose Estimator", layout="wide")

# --------------------------
# Utility: RGB to HSV
# --------------------------
def rgb_to_hsv(rgb):
    """Convert Nx3 array of RGB [0-1] to HSV [0-1]"""
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
# Bubble feature extraction
# --------------------------
def extract_bubble_features(image_path, visualize=True, top_n=20):
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
        r = int(r * 0.9)  # shrink to 90%
        Y, X = np.ogrid[:img_rgb.shape[0], :img_rgb.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        roi_rgb = img_rgb[mask]
        roi_hsv = rgb_to_hsv(roi_rgb / 255.0)
        h_mean, s_mean, v_mean = roi_hsv.mean(axis=0)

        # HSV pink filter
        if 252/360 <= h_mean <= 290/360 and s_mean >= 0.07 and v_mean >= 0.60:
            score = (h_mean**8) * r
            candidates.append({
                "x": int(x),
                "y": int(y),
                "r": r,
                "score": score,
                "roi_hsv": roi_hsv
            })

    if len(candidates) == 0:
        raise ValueError("No bubbles passed HSV filter.")

    # Sort & select top non-overlapping
    candidates = sorted(candidates, key=lambda b: b["score"], reverse=True)
    selected = []
    for b in candidates:
        overlap = any(np.sqrt((b["x"]-sb["x"])**2 + (b["y"]-sb["y"])**2) < (b["r"]+sb["r"]) for sb in selected)
        if not overlap:
            selected.append(b)
        if len(selected) >= top_n:
            break

    avg_hsv = np.mean([b["roi_hsv"].mean(axis=0) for b in selected], axis=0)

    if visualize:
        img_vis = img_rgb.copy()
        for b in selected:
            cv2.circle(img_vis, (b["x"], b["y"]), b["r"], (255,0,0), 2)
        st.image(img_vis, caption=f"Top {len(selected)} selected bubbles", use_column_width=True)

    return avg_hsv

# --------------------------
# Calibration data
# --------------------------
calibration_data = pd.DataFrame({
    "Glucose": [25, 50, 75, 100, 125],  # µM
    "H": [0.722, 0.733, 0.740, 0.730, 0.786],
    "S": [0.087, 0.092, 0.102, 0.092, 0.113]
})

# Baseline from blank saliva
H_blank_deg = 215
S_blank_percent = 8.2
H_blank = H_blank_deg / 360.0
S_blank = S_blank_percent / 100.0

# Correct calibration
calibration_data["H_corr"] = calibration_data["H"] - H_blank
calibration_data["S_corr"] = calibration_data["S"] - S_blank
y_glucose = calibration_data["Glucose"].values

model_H  = LinearRegression().fit(calibration_data[["H_corr"]], y_glucose)
model_S  = LinearRegression().fit(calibration_data[["S_corr"]], y_glucose)
model_HS = LinearRegression().fit(calibration_data[["H_corr","S_corr"]], y_glucose)

# --------------------------
# Streamlit UI
# --------------------------
st.title("Saliva Glucose Estimator")
uploaded_file = st.file_uploader("Upload a photo of your saliva bubbles", type=["jpg","png","jpeg"])

if uploaded_file:
    # Save temp file
    temp_path = os.path.join("temp_image.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        avg_hsv = extract_bubble_features(temp_path)
        H_avg, S_avg, V_avg = avg_hsv

        # Baseline-corrected
        H_corr_input = H_avg - H_blank
        S_corr_input = max(S_avg - S_blank, 0.0)  # clamp at 0

        df_H  = pd.DataFrame({"H_corr":[H_corr_input]})
        df_S  = pd.DataFrame({"S_corr":[S_corr_input]})
        df_HS = pd.DataFrame({"H_corr":[H_corr_input], "S_corr":[S_corr_input]})

        g_H  = max(model_H.predict(df_H)[0], 0)
        g_S  = max(model_S.predict(df_S)[0], 0)
        g_HS = max(model_HS.predict(df_HS)[0], 0)

        glucose_avg = 0.6*g_H + 0.4*g_S

        st.subheader("Estimated Glucose (µM)")
        st.write(f"H only: {g_H:.1f}")
        st.write(f"S only: {g_S:.1f}")
        st.write(f"H + S multivariate: {g_HS:.1f}")
        st.write(f"Weighted average: {glucose_avg:.1f} µM")

    except Exception as e:
        st.error(f"Error processing image: {e}")

