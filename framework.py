import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(page_title="Forgery Dataset Feature Extractor", layout="wide")
st.title("✍ Forged Handwritten Document Database - Parent Folder Feature Extractor")

# ---------------- Feature Extraction ---------------- #
def extract_features(image_path, class_label, dataset_name):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "image_path": image_path,
                "file_name": os.path.basename(image_path),
                "class": class_label,
                "dataset": dataset_name,
                "error": "Unreadable file"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # KB
        aspect_ratio = round(width / height, 3)

        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "image_path": image_path,
            "file_name": os.path.basename(image_path),
            "class": class_label,
            "dataset": dataset_name,
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
        return {"image_path": image_path, "file_name": os.path.basename(image_path),
                "class": class_label, "dataset": dataset_name, "error": str(e)}

# ---------------- Parent Folder Input ---------------- #
parent_folder = st.text_input("📂 Enter parent dataset folder path:")

all_records = []

if parent_folder and os.path.isdir(parent_folder):
    dataset_name = os.path.basename(parent_folder)
    classes = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    st.success(f"✅ Found {len(classes)} classes → {classes}")

    for class_dir in classes:
        class_path = os.path.join(parent_folder, class_dir)
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

        st.write(f"📂 Class '{class_dir}' → {len(files)} images")
        for fname in files:
            path = os.path.join(class_path, fname)
            rec = extract_features(path, class_dir, dataset_name)
            all_records.append(rec)

# ---------------- Results ---------------- #
if all_records:
    df = pd.DataFrame(all_records)
    st.subheader("📊 Extracted Features (Preview)")
    st.dataframe(df.head(20))

    save_path = os.path.join(parent_folder, "metadata_features.csv")
    df.to_csv(save_path, index=False)
    st.success(f"💾 Features saved to {save_path}")

    if "class" in df.columns:
        st.subheader("📈 Class Distribution")
        st.bar_chart(df["class"].value_counts())

    st.subheader("🖼 Sample Images")
    cols = st.columns(5)
    grouped = df.groupby("class").first().reset_index()

    for idx, row in grouped.iterrows():
        try:
            img_path = row["image_path"]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                cols[idx % 5].image(img, caption=row["class"], use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load image for {row['class']}: {e}")
