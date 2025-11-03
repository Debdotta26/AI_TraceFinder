import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy


st.set_page_config(page_title="Forgery Dataset Feature Extractor", layout="wide")
st.title("✍️ Forged Handwritten Document Database - Auto Class Detection & Feature Extraction")


def extract_features(image_path, class_label):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "file_name": os.path.basename(image_path),
                "class_label": class_label,
                "error": "Unreadable file"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Basic shape + file info
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)

        # Stats
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "file_name": os.path.basename(image_path),
            "class_label": class_label,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3)
        }
    except Exception as e:
        return {
            "file_name": image_path,
            "class_label": class_label,
            "error": str(e)
        }


dataset_root = st.text_input(" Enter dataset root path:", "")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == ".":
            continue  # skip the root itself

        # Full relative folder path becomes the class label
        class_label = rel_path.replace(os.sep, "/")

        # Collect image files
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

        # ✅ Only show folders that actually contain images
        if len(image_files) > 0:
            st.write(f" Class '{class_label}' → {len(image_files)} images")

        for fname in image_files:
            path = os.path.join(root, fname)
            rec = extract_features(path, class_label)
            records.append(rec)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    st.subheader(" Features Extracted (Preview)")
    st.dataframe(df.head(20))

    # Save features
    save_path = os.path.join(dataset_root, "metadata_features.csv")
    df.to_csv(save_path, index=False)
    st.success(f" Features saved to {save_path}")

    # Class distribution
    if "class_label" in df.columns:
        st.subheader(" Class Distribution")
        st.bar_chart(df["class_label"].value_counts())

    # Sample Images
    st.subheader(" Sample Images")
    cols = st.columns(5)
    shown_classes = set()
