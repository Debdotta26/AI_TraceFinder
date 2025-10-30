# main_app.py
import streamlit as st
import pandas as pd
import joblib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import random
import pickle
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# Supported image extensions
EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

CSV_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\results\\metadata_features.csv"

# Hybrid model artifact paths (used for evaluation & prediction)
HYB_ART_DIR = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models"
HYB_MODEL_PATH = os.path.join(HYB_ART_DIR, "scanner_hybrid_final.keras")
HYB_ENCODER_PATH = os.path.join(HYB_ART_DIR, "hybrid_label_encoder.pkl")
HYB_SCALER_PATH = os.path.join(HYB_ART_DIR, "hybrid_feat_scaler.pkl")
HYB_FP_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models\\scanner_fingerprints.pkl"  # as in your scripts
HYB_ORDER_NPY = os.path.join(HYB_ART_DIR, "fp_keys.npy")
HYB_RES_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\new_processed\\official_wiki_residuals.pkl"
HYB_HISTORY_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models\\hybrid_training_history.pkl"

# Ensure plotting looks okay in Streamlit
sns.set_theme()


import os
import cv2
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from tensorflow import keras
from skimage.feature import local_binary_pattern as sk_lbp
from scipy.stats import skew, kurtosis, entropy
from scipy.ndimage import sobel

# ================================
# ==== FILE PATH CONFIG ==========
# ================================
HYB_MODEL_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models\\scanner_hybrid.keras"
HYB_ENCODER_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models\\hybrid_label_encoder.pkl"
HYB_SCALER_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\hybrid_models\\hybrid_feat_scaler.pkl"
HYB_RES_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\official_wiki_residuals.pkl"


def prediction():
    st.title("🔍 Hybrid Model Prediction")
    st.write("Upload an image to predict the scanner model using the hybrid CNN model.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    # ================================
    # ==== IMAGE UTILITIES ===========
    # ================================
    def load_and_preprocess(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image.")
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        return img

    # ================================
    # ==== FEATURE EXTRACTION ========
    # ================================
    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K + 1)
        feats = [
            float(np.mean(mag[(r >= bins[i]) & (r < bins[i + 1])]))
            if ((r >= bins[i]) & (r < bins[i + 1])).any()
            else 0.0
            for i in range(K)
        ]
        return feats

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255).astype(np.uint8)
        codes = sk_lbp(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P + 3), density=True)
        return hist.astype(np.float32).tolist()

    def compute_all_features(img):
        lbp_feats = lbp_hist_safe(img)      # 10 features
        fft_feats = fft_radial_energy(img)  # 6 features

        mean = np.mean(img)
        std = np.std(img)
        skw = skew(img.ravel())
        kurt = kurtosis(img.ravel())
        hist, _ = np.histogram(img.ravel(), bins=256, density=True)
        entr = entropy(hist + 1e-10)
        sob = sobel(img)
        sob_mean = np.mean(sob)
        sob_std = np.std(sob)
        sob_max = np.max(sob)
        sob_min = np.min(sob)
        contrast = np.var(sob)
        energy = np.sum(img ** 2)

        feats = lbp_feats + fft_feats + [
            mean, std, skw, kurt, entr,
            sob_mean, sob_std, sob_max, sob_min, contrast, energy
        ]
        return feats

   # ================================
# ==== HYBRID MODEL PREDICTION ===
# ================================
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

def predict_scanner_hybrid(img_path):
    # === Define model & artifact paths ===
    base_dir = "C:\\Users\\ariji\\Desktop\\ai_trace"

    # === Validate paths ===
    if not os.path.exists(HYB_MODEL_PATH):
        raise FileNotFoundError(f"❌ Hybrid model not found at {HYB_MODEL_PATH}")
    if not os.path.exists(HYB_ENCODER_PATH):
        raise FileNotFoundError(f"❌ Label encoder not found at {HYB_ENCODER_PATH}")
    if not os.path.exists(HYB_SCALER_PATH):
        raise FileNotFoundError(f"❌ Scaler not found at {HYB_SCALER_PATH}")

    # === Load model and preprocessing tools ===
    model = keras.models.load_model(HYB_MODEL_PATH, compile=False)
    with open(HYB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(HYB_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # === Preprocess the image ===
    img = load_and_preprocess(img_path)
    feats = compute_all_features(img)
    X_scaled = scaler.transform([feats])
    img_input = np.expand_dims(img, axis=(0, -1))

    # === Predict ===
    num_inputs = len(model.inputs)
    if num_inputs == 2:
        probs = model.predict([img_input, X_scaled], verbose=0)[0]
    else:
        probs = model.predict(X_scaled, verbose=0)[0]

    # === Postprocess results ===
    idx = np.argmax(probs)
    pred_label = le.inverse_transform([idx])[0]
    confidence = probs[idx] * 100

    top_k_idx = np.argsort(probs)[::-1]
    top_labels = le.inverse_transform(top_k_idx)
    top_probs = probs[top_k_idx] * 100

    df = pd.DataFrame({
        "Scanner Model": top_labels,
        "Probability (%)": top_probs.round(3)
    })

    return pred_label, confidence, df




# === TRAIN CLASSICAL MODELS (RF + SVM) ===
def train_models():
    st.subheader("Train Random Forest & SVM")
    st.info("This trains Random Forest and SVM using the metadata features CSV.")

    if st.button("🚀 Start Training (RF + SVM)"):
        if not os.path.exists(CSV_PATH):
            st.error(f"CSV file not found at {CSV_PATH}")
            return

        os.makedirs("models", exist_ok=True)
        df = pd.read_csv(CSV_PATH)
        X = df.drop(columns=["file_name", "class_label"])
        y = df["class_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train_scaled, y_train)
        joblib.dump(rf, "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\random_forest.pkl")

        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        svm.fit(X_train_scaled, y_train)
        joblib.dump(svm, "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\svm.pkl")

        joblib.dump(scaler, "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\scaler.pkl")

        # Evaluate quickly and show metrics
        X_test_scaled = scaler.transform(X_test)
        rf_pred = rf.predict(X_test_scaled)
        svm_pred = svm.predict(X_test_scaled)

        st.markdown("**Random Forest (test split)**")
        st.text(classification_report(y_test, rf_pred))
        st.markdown("**SVM (test split)**")
        st.text(classification_report(y_test, svm_pred))

        st.success("✅ RF and SVM trained and saved to `models/`.")


# === EVALUATE CLASSICAL MODEL (from metadata CSV) ===
def evaluate_model(model_path, name):
    if not os.path.exists(model_path):
        st.warning(f"{name} model not found at {model_path}")
        return

    if not os.path.exists(CSV_PATH):
        st.warning(f"Metadata CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "class_label"])
    y = df["class_label"]

    scaler = joblib.load("C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\scaler.pkl")
    model = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader(f"{name} Classification Report")
    st.text(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix")
    st.pyplot(fig)


# === IMAGE PREPROCESSING UTILITIES (used across app) ===
def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

# === PREDICTION: RF / SVM ===
def predict_scanner(img_path, model_choice="rf"):
    scaler = joblib.load("C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\scaler.pkl")
    model = joblib.load(f"models/{'random_forest' if model_choice == 'rf' else 'svm'}.pkl")

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    return pred, prob


# === HYBRID PREDICTION (loads hybrid model artifacts) ===
def predict_scanner_hybrid(img_path):
    import tensorflow as tf
    from tensorflow import keras
    import pickle, numpy as np, os
    from skimage.feature import local_binary_pattern as sk_lbp

    # --- Check all files exist ---
    for p in [HYB_MODEL_PATH, HYB_ENCODER_PATH, HYB_SCALER_PATH, HYB_FP_PATH, HYB_ORDER_NPY]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # --- Load all artifacts ---
    model = keras.models.load_model(HYB_MODEL_PATH, compile=False)
    with open(HYB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(HYB_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(HYB_FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(HYB_ORDER_NPY, allow_pickle=True).tolist()

    # === Feature utilities ===
    def corr2d(a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / denom) if denom != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        bins = np.linspace(0, r.max() + 1e-6, K + 1)
        feats = [
            float(np.mean(mag[(r >= bins[i]) & (r < bins[i + 1])])) if ((r >= bins[i]) & (r < bins[i + 1])).any() else 0.0
            for i in range(K)
        ]
        return feats

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g * 255).astype(np.uint8)
        codes = sk_lbp(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P + 3), density=True)
        return hist.astype(np.float32).tolist()

    # --- Preprocess image ---
    img = load_and_preprocess(img_path, size=(256, 256))

    # --- Compute handcrafted features ---
    v_corr = [corr2d(img, scanner_fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(img, K=6)
    v_lbp = lbp_hist_safe(img)
    features = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)

    # --- Prepare CNN and feature inputs ---
    X_img = np.expand_dims(img, axis=(0, -1))
    X_feat = scaler.transform(features)

    # --- Predict (ensure correct input order) ---
    try:
        probs = model.predict([X_img, X_feat], verbose=0)[0]
    except Exception:
        probs = model.predict([X_feat, X_img], verbose=0)[0]

    pred_class = le.inverse_transform([int(np.argmax(probs))])[0]
    conf = float(np.max(probs)) * 100.0

    st.success(f"🧾 Prediction: **{pred_class}** ({conf:.2f}% confidence)")
    return pred_class, conf



# === PREPROCESSING (kept as you wrote it) ===
def run_preprocessing():
    import cv2, pywt, pickle, numpy as np, os
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scipy.signal import wiener as scipy_wiener
    from skimage.restoration import denoise_wavelet

    st.subheader("🧪 Preprocess Datasets")
    st.info("This will generate residuals for Official, Wikipedia, and Flatfield datasets.")

    IMG_SIZE = (256, 256)
    DENOISE_METHOD = "wavelet"
    MAX_WORKERS = 8
    BASE_DIR = "C:\\Users\\ariji\\Desktop\\ai_trace\\process_data"

    def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    def resize_to(img): return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    def normalize_img(img): return img.astype(np.float32) / 255.0
    def denoise_wavelet_img(img):
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        cH[:] = 0; cV[:] = 0; cD[:] = 0
        return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    def preprocess_image(fpath, method=DENOISE_METHOD):
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        img = to_gray(img)
        img = resize_to(img)
        img = normalize_img(img)
        den = scipy_wiener(img, mysize=(5,5)) if method == "wiener" else denoise_wavelet_img(img)
        return (img - den).astype(np.float32)

    def parallel_process_images(file_list):
        residuals = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(preprocess_image, f) for f in file_list]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    residuals.append(res)
        return residuals

    def process_folder(base_dir, use_dpi_subfolders=True):
        residuals_dict = {}
        scanners = sorted(os.listdir(base_dir))
        for scanner in scanners:
            scanner_dir = os.path.join(base_dir, scanner)
            if not os.path.isdir(scanner_dir): continue
            residuals_dict[scanner] = {}
            if use_dpi_subfolders:
                for dpi in os.listdir(scanner_dir):
                    dpi_dir = os.path.join(scanner_dir, dpi)
                    if not os.path.isdir(dpi_dir): continue
                    files = [os.path.join(dpi_dir, f) for f in os.listdir(dpi_dir)
                             if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
                    residuals_dict[scanner][dpi] = parallel_process_images(files)
            else:
                files = [os.path.join(scanner_dir, f) for f in os.listdir(scanner_dir)
                         if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
                residuals_dict[scanner] = parallel_process_images(files)
        return residuals_dict

    def save_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        st.write(f"✅ Saved to {path}")

    if st.button("🚀 Run Preprocessing"):
        st.write("🔄 Processing Official and Wikipedia datasets...")
        official_wiki_residuals = {}
        for dataset in ["Official", "Wikipedia"]:
            dataset_dir = os.path.join(BASE_DIR, dataset)
            official_wiki_residuals[dataset] = process_folder(dataset_dir, use_dpi_subfolders=True)
        save_pickle(official_wiki_residuals, os.path.join(BASE_DIR, "official_wiki_residuals.pkl"))

        st.write("🔄 Processing Flatfield dataset...")
        flatfield_dir = os.path.join(BASE_DIR, "Flatfield")
        flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)
        save_pickle(flatfield_residuals, os.path.join(BASE_DIR, "flatfield_residuals.pkl"))

        total_scanners = len(flatfield_residuals)
        total_images = sum(len(v) for v in flatfield_residuals.values())
        st.success(f"✅ Done. Flatfield: {total_scanners} scanners, {total_images} images")

def train_hybrid_model():
    import tensorflow as tf
    from tensorflow import keras
    import pickle, numpy as np, io, cv2, os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from contextlib import redirect_stdout
    import streamlit as st

    st.subheader("🧠 Hybrid Model Management")

    required_files = [
        HYB_MODEL_PATH, HYB_ENCODER_PATH, HYB_SCALER_PATH,
        HYB_FP_PATH, HYB_ORDER_NPY, HYB_HISTORY_PATH  # include history path
    ]

    if not all(os.path.exists(p) for p in required_files):
        st.error("❌ Hybrid model not found. Please train it first.")
        st.info("Missing files:")
        for p in required_files:
            if not os.path.exists(p):
                st.write(f"🚫 {p}")
        return None, None, None, None, None, None

    # --- Load saved model and components ---
    st.success("✅ Loading trained hybrid model and artifacts...")

    model = keras.models.load_model(HYB_MODEL_PATH, compile=False)
    with open(HYB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(HYB_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(HYB_FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(HYB_ORDER_NPY, allow_pickle=True).tolist()
    with open(HYB_HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    st.info("✅ Model and components loaded successfully.")

    # --- Show model summary ---
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()
    summary_str = buffer.getvalue()
    st.text_area("Model Summary", summary_str, height=300)

    # --- Display info ---
    st.write("📂 Number of scanner fingerprints:", len(scanner_fps))
    st.write("🧩 Feature scaler shape:", getattr(scaler, "mean_", "N/A"))
    st.write("🏷️ Classes:", list(le.classes_))

    # --- Display few fingerprints safely ---
    st.subheader("🔍 Sample Fingerprints")
    if len(scanner_fps) > 0:
        sample_keys = list(scanner_fps.keys())[:3]
        for k in sample_keys:
            img = scanner_fps[k]
            img = cv2.resize(img, (256, 256))
            img = np.clip(img / 255.0, 0, 1)
            st.image(img, caption=f"Fingerprint: {k}", use_column_width=True, clamp=True)
    else:
        st.warning("⚠️ No fingerprints found in the loaded data.")

    # --- Accuracy and Loss Curves ---
    st.subheader("📈 Training Curves")

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        ax_acc.plot(history['val_accuracy'], label='Val Accuracy')
    ax_acc.set_title('Accuracy over Epochs')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    st.pyplot(fig_acc)

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        ax_loss.plot(history['val_loss'], label='Val Loss')
    ax_loss.set_title('Loss over Epochs')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    st.pyplot(fig_loss)

    # --- Confusion Matrix from PNG ---
    cm_path = "C:\\Users\\ariji\\Desktop\\ai_trace\\results\\hybrid cnn.png"
    if os.path.exists(cm_path):
        st.subheader("📊 Confusion Matrix")
        st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.error("❌ Confusion matrix file not found in result folder.")

    st.success("✅ Model ready for prediction.")
    return model, le, scaler, scanner_fps, fp_keys, cm_path
# === FEATURE EXPLORER ===
def feature_explorer():
    if not os.path.exists(CSV_PATH):
        st.warning(f"Metadata CSV not found at {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    st.subheader("📊 Feature Explorer")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        feature = st.selectbox("Choose feature to visualize", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)


# === DATA VISUALIZATION (EDA) ===
import streamlit as st
import os, cv2, random, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def streamlit_eda():
    st.title("📊 Data Visualization & EDA")
    st.markdown("### Analyze dataset structure, class balance, and sample quality")

    # === Dataset selection ===
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Official", "Wikipedia", "Flatfield"],
        index=0
    )
    dataset_map = {
        "Official": r"C:\Users\ariji\Desktop\ai_trace\process_data\Official",
        "Wikipedia": r"C:\Users\ariji\Desktop\ai_trace\process_data\Wikipedia",
        "Flatfield": r"C:\Users\ariji\Desktop\ai_trace\process_data\Flatfield"
    }
    dataset_path = dataset_map[dataset_choice]

    if not os.path.exists(dataset_path):
        st.warning(f"⚠️ Dataset path not found: `{dataset_path}`")
        return

    # === Initialize ===
    class_counts = {}
    corrupted_files, duplicates = [], []
    image_shapes, brightness_values = [], []
    hashes, valid_images = {}, []

    # All top-level class folders
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    total_classes = len(class_dirs)
    progress = st.progress(0)
    st.info(f"🔍 Scanning `{dataset_choice}` dataset (recursive)...")

    # === Recursive scan ===
    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(dataset_path, class_name)
        count = 0

        # Walk through nested subfolders
        for root, _, files in os.walk(class_path):
            for f in files:
                if os.path.splitext(f)[1].lower() not in EXTENSIONS:
                    continue

                file_path = os.path.join(root, f)
                img = cv2.imread(file_path)
                if img is None:
                    corrupted_files.append(file_path)
                    continue

                h, w = img.shape[:2]
                image_shapes.append((h, w))
                brightness_values.append(img.mean())
                valid_images.append(file_path)

                # Hash to detect duplicates
                try:
                    with open(file_path, "rb") as f_img:
                        img_hash = hashlib.md5(f_img.read()).hexdigest()
                    if img_hash in hashes:
                        duplicates.append(file_path)
                    else:
                        hashes[img_hash] = file_path
                except Exception:
                    continue

                count += 1

        class_counts[class_name] = count
        progress.progress((idx + 1) / total_classes)

    # === Summary Table ===
    if class_counts:
        df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class Name", "Image Count"])
        df_counts["% of Total"] = (df_counts["Image Count"] / df_counts["Image Count"].sum() * 100).round(2)
        st.markdown("### 🧾 Class-wise Image Count (Recursive)")
        st.dataframe(df_counts, use_container_width=True)

    # === Metrics Overview ===
    total_images = sum(class_counts.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_images)
    col2.metric("Corrupted Images", len(corrupted_files))
    col3.metric("Duplicate Images", len(duplicates))

    # === Class Distribution Chart ===
    if class_counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(class_counts.keys(), class_counts.values(), color="#4C9AFF")
        ax.set_ylabel("Image Count")
        ax.set_title(f"Class Distribution - {dataset_choice}")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # === Aspect Ratio & Brightness Stats ===
    if image_shapes:
        heights, widths = zip(*image_shapes)
        aspect_ratios = [w / h for h, w in image_shapes]

        col4, col5 = st.columns(2)
        with col4:
            fig2, ax2 = plt.subplots()
            ax2.hist(aspect_ratios, bins=20, color="#34C759")
            ax2.set_xlabel("Aspect Ratio (W/H)")
            ax2.set_title("Aspect Ratio Distribution")
            st.pyplot(fig2)

        with col5:
            fig3, ax3 = plt.subplots()
            ax3.hist(brightness_values, bins=30, color="#FF9500")
            ax3.set_xlabel("Mean Brightness")
            ax3.set_title("Brightness Distribution")
            st.pyplot(fig3)

    # === Corrupted / Duplicates ===
    if corrupted_files or duplicates:
        st.markdown("### ⚠️ Data Quality Issues")
        if corrupted_files:
            st.error(f"🧨 {len(corrupted_files)} corrupted images found.")
        if duplicates:
            st.warning(f"♻️ {len(duplicates)} duplicate images detected.")

    # === Random Samples per Class ===
    st.markdown("### 🎯 Random Samples from Each Class")
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        all_images = []
        for root, _, files in os.walk(class_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in EXTENSIONS:
                    all_images.append(os.path.join(root, f))
        if not all_images:
            continue

        sample_files = random.sample(all_images, min(3, len(all_images)))
        st.markdown(f"#### 📁 {class_name} ({len(all_images)} images)")
        cols = st.columns(len(sample_files))
        for col, fpath in zip(cols, sample_files):
            try:
                img = Image.open(fpath)
                col.image(img, caption=os.path.basename(fpath), use_column_width=True)
            except Exception:
                col.error("⚠️ Failed to load image")

    st.success("✅ Recursive EDA Completed Successfully!")

# === ENHANCED FEATURE EXTRACTION ===
def run_feature_extraction():
    import pickle, numpy as np
    from tqdm import tqdm
    from skimage.feature import local_binary_pattern
    from scipy import ndimage
    from scipy.fft import fft2, fftshift

    st.subheader("📊 Feature Extraction")
    st.info("Running PRNU and enhanced feature extraction...")

    FLATFIELD_RESIDUALS_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\new_processed\\flatfield_residuals.pkl"
    FP_OUT_PATH = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\new_processed\\scanner_fingerprints.pkl"
    ORDER_NPY = os.path.join(HYB_ART_DIR, "fp_keys.npy")
    RES_PATH_LOCAL = "C:\\Users\\ariji\\Desktop\\ai_trace\\models\\new_processed\\official_wiki_residuals.pkl"
    FEATURES_OUT = os.path.join(HYB_ART_DIR, "features.pkl")
    ENHANCED_OUT = os.path.join(HYB_ART_DIR, "enhanced_features.pkl")

    if not os.path.exists(FLATFIELD_RESIDUALS_PATH):
        st.error(f"Flatfield residuals not found at {FLATFIELD_RESIDUALS_PATH}")
        return

    with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
        flatfield_residuals = pickle.load(f)

    scanner_fingerprints = {}
    for scanner, residuals in flatfield_residuals.items():
        if residuals:
            stack = np.stack(residuals, axis=0)
            fingerprint = np.mean(stack, axis=0)
            scanner_fingerprints[scanner] = fingerprint

    with open(FP_OUT_PATH, "wb") as f:
        pickle.dump(scanner_fingerprints, f)

    fp_keys = sorted(scanner_fingerprints.keys())
    np.save(ORDER_NPY, np.array(fp_keys))

    def corr2d_local(a, b):
        a = a.astype(np.float32).ravel()
        b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / denom) if denom != 0 else 0.0

    with open(RES_PATH_LOCAL, "rb") as f:
        residuals_dict = pickle.load(f)

    features, labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    vec = [corr2d_local(res, scanner_fingerprints[k]) for k in fp_keys]
                    features.append(vec)
                    labels.append(scanner)

    with open(FEATURES_OUT, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)

    def extract_enhanced_features(residual):
        fft_img = np.abs(fft2(residual))
        fft_img = fftshift(fft_img)
        h, w = fft_img.shape
        ch, cw = h//2, w//2
        low = np.mean(fft_img[ch-20:ch+20, cw-20:cw+20])
        mid = np.mean(fft_img[ch-60:ch+60, cw-60:cw+60]) - low
        high = np.mean(fft_img) - low - mid

        lbp = local_binary_pattern(residual, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0,25), density=True)

        grad_x = ndimage.sobel(residual, axis=1)
        grad_y = ndimage.sobel(residual, axis=0)
        grad_mag = np.sqrt(grad_x*2 + grad_y*2)
        texture = [np.std(residual), np.mean(np.abs(residual)), np.std(grad_mag), np.mean(grad_mag)]

        return [low, mid, high] + lbp_hist.tolist() + texture

    enhanced_features, enhanced_labels = [], []
    for dataset_name in ["Official", "Wikipedia"]:
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    feat = extract_enhanced_features(res)
                    enhanced_features.append(feat)
                    enhanced_labels.append(scanner)

    with open(ENHANCED_OUT, "wb") as f:
        pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)

    st.success("✅ Feature extraction complete!")
    st.write(f"PRNU features saved: {len(features)} samples")
    st.write(f"Enhanced features saved: {len(enhanced_features)} samples")


# === EVALUATE HYBRID MODEL (full pipeline) ===
def evaluate_hybrid_model():
    import tensorflow as tf
    from tensorflow import keras
    from skimage.feature import local_binary_pattern as sk_lbp

    st.subheader("🧠 Evaluate Hybrid CNN")

    # Confirm artifacts exist
    missing = []
    for p in [HYB_MODEL_PATH, HYB_ENCODER_PATH, HYB_SCALER_PATH, HYB_FP_PATH, HYB_ORDER_NPY, HYB_RES_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        st.error("Missing hybrid artifacts. Please ensure these files exist:\n" + "\n".join(missing))
        return

    # Load artifacts
    with open(HYB_FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(HYB_ORDER_NPY, allow_pickle=True).tolist()
    model = tf.keras.models.load_model(HYB_MODEL_PATH, compile=False)
    with open(HYB_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(HYB_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Utilities (same as training)
    def corr2d(a, b):
        a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
        a -= a.mean(); b -= b.mean()
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        return float((a @ b) / denom) if denom != 0 else 0.0

    def fft_radial_energy(img, K=6):
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f)
        h, w = mag.shape; cy, cx = h//2, w//2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6
        bins = np.linspace(0, rmax, K+1)
        feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]
        return feats

    def lbp_hist_safe(img, P=8, R=1.0):
        rng = float(np.ptp(img))
        g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
        g8 = (g*255).astype(np.uint8)
        codes = sk_lbp(g8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
        return hist.astype(np.float32).tolist()

    # Load residuals for evaluation
    with open(HYB_RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    X_img_te, X_feat_te, y_te = [], [], []
    for dataset_name in ["Official", "Wikipedia"]:
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    X_img_te.append(np.expand_dims(res,-1))
                    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                    v_fft  = fft_radial_energy(res)
                    v_lbp  = lbp_hist_safe(res)
                    X_feat_te.append(v_corr + v_fft + v_lbp)
                    y_te.append(scanner)

    X_img_te = np.array(X_img_te, dtype=np.float32)
    X_feat_te = np.array(X_feat_te, dtype=np.float32)
    y_int_te = np.array([le.transform([c])[0] for c in y_te])

    X_feat_te = scaler.transform(X_feat_te)

    st.info("Running hybrid model predictions (this may take a while)...")
    y_pred_prob = model.predict([X_img_te, X_feat_te])
    y_pred = np.argmax(y_pred_prob, axis=1)

    test_acc = accuracy_score(y_int_te, y_pred)
    st.success(f"✅ Hybrid Test Accuracy: {test_acc*100:.2f}%")
    st.text("✅ Classification Report:")
    st.text(classification_report(y_int_te, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_int_te, y_pred)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Hybrid CNN Confusion Matrix")
    st.pyplot(fig)

import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy


def extract_features(image_path, class_label):
    """Extract statistical, texture, and edge-based features from an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "file_name": os.path.basename(image_path),
                "class_label": class_label,
                "error": "Unreadable file"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  # in KB
        aspect_ratio = round(width / height, 3)

        # Statistical features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        # Edge density
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


def baseline_feature_extraction(dataset_root):
    """
    Scans the dataset directory and extracts features from all image files.
    Returns the resulting DataFrame and also saves it as 'metadata_features.csv'.
    """
    if not dataset_root or not os.path.isdir(dataset_root):
        raise ValueError("Invalid dataset root path.")

    records = []

    for root, dirs, files in os.walk(dataset_root):
        rel_path = os.path.relpath(root, dataset_root)
        if rel_path == ".":
            continue

        class_label = rel_path.replace(os.sep, "/")

        image_files = [
            f for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]

        for fname in image_files:
            path = os.path.join(root, fname)
            rec = extract_features(path, class_label)
            records.append(rec)

    df = pd.DataFrame(records)
    save_path = os.path.join(dataset_root, "metadata_features.csv")
    df.to_csv(save_path, index=False)

    return df, save_path

def evaluate_baseline_models(metadata_csv, models_dir):
    """
    Loads baseline models (joblib or pickle) and evaluates them on features from metadata CSV.
    Returns evaluation results and plots.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError("metadata_features.csv not found. Run feature extraction first.")

    df = pd.read_csv(metadata_csv)
    if "class_label" not in df.columns:
        raise ValueError("CSV must contain 'class_label' column.")

    # Separate features and labels
    X = df.drop(columns=["file_name", "class_label", "error"], errors="ignore")
    y = df["class_label"]

    # Load all models in models_dir
    model_files = [
        f for f in os.listdir(models_dir)
        if f.lower().endswith((".joblib", ".pkl"))
    ]
    results = []

    for mf in model_files:
        model_path = os.path.join(models_dir, mf)
        try:
            model = joblib.load(model_path)
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            cm = confusion_matrix(y, preds)
            cls_report = classification_report(y, preds, output_dict=True)
            results.append({
                "model": mf,
                "accuracy": acc,
                "report": cls_report,
                "confusion_matrix": cm
            })
        except Exception as e:
            results.append({
                "model": mf,
                "error": str(e)
            })

    return results



# === STREAMLIT UI ===
## =========================================
# === STREAMLIT MAIN UI (APP ENTRYPOINT) ===
# =========================================

st.set_page_config(page_title="AI TraceFinder", layout="wide")

# === SIDEBAR MENU ===
with st.sidebar:
    st.sidebar.title("AI TraceFinder")
    st.sidebar.caption("Digital Scanner Forensics")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Choose Action", [
    "HOME",
    "📊 Data Visualization",
    "📊 Feature Explorer",
    "🔍 Prediction",
    "🧪 Preprocess Datasets",
    "🧠 Hybrid Model Management",
    "🧩 Baseline Feature Extractor",
    "📈 Baseline Model Report"   # 👈 Add this new option
])


# === MENU HANDLERS ===
if menu == "HOME":
    st.title("Welcome to AI TraceFinder 🔍")
    st.write("""
    This tool identifies the **scanner model** of a given scanned image using a 
    **Hybrid CNN model** combining handcrafted and deep features.
    """)

elif menu == "📊 Data Visualization":
    streamlit_eda()

elif menu == "📊 Feature Explorer":
    feature_explorer()

elif menu == "🧪 Preprocess Datasets":
    run_preprocessing()

elif menu == "🧠 Train Models":
    train_hybrid_model()

elif menu == "📈 Evaluate Models":
    st.subheader("📈 Evaluate Models")

    with st.expander("📊 Evaluate Random Forest", expanded=True):
        evaluate_model("C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\random_forest.pkl", "Random Forest")

    with st.expander("📊 Evaluate SVM", expanded=False):
        evaluate_model("C:\\Users\\ariji\\Desktop\\ai_trace\\models\\baseline_models\\svm.pkl", "SVM")

    with st.expander("🧠 Evaluate Hybrid CNN", expanded=False):
        if st.button("🚀 Run Hybrid Evaluation"):
            evaluate_hybrid_model()

elif menu == "🔍 Predict Scanner":
    st.subheader("Upload Image for Prediction")

    uploaded_file = st.file_uploader("Upload an Image", type=["tif", "tiff", "png", "jpg", "jpeg"])
    model_choice = st.selectbox("Choose Model", ["Random Forest", "SVM", "Hybrid CNN"])

    if uploaded_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(temp_path, caption="Uploaded Image", use_container_width=True)
        st.markdown("---")

        try:
            if model_choice == "Hybrid CNN":
                pred_label, confidence, df = predict_scanner_hybrid(temp_path)

                # === BEAUTIFUL DISPLAY SECTION ===
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:20px; border-radius:12px; 
                                background-color:#f8f9fa; box-shadow:0 0 10px rgba(0,0,0,0.1);">
                        <h3 style="color:#2c3e50;">🧠 Predicted Scanner Model: 
                            <span style="color:#16a085;">{pred_label}</span></h3>
                        <h4 style="color:#555;">Confidence: <b>{confidence:.2f}%</b></h4>
                        <p style="background-color:#e3f2fd; padding:8px; border-radius:8px; 
                                  display:inline-block; color:#1565c0;">
                            Prediction made using <b>Hybrid CNN Model</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🔝 Top Prediction Probability")
                st.dataframe(df.head(10), use_container_width=True)

            else:
                # === CLASSICAL ML MODELS (RF / SVM) ===
                pred, prob = predict_scanner(
                    temp_path,
                    model_choice="rf" if model_choice == "Random Forest" else "svm"
                )

                st.markdown(
                    f"""
                    <div style="text-align:center; padding:20px; border-radius:12px; 
                                background-color:#f9f9f9; box-shadow:0 0 10px rgba(0,0,0,0.05);">
                        <h3 style="color:#2c3e50;">🖼 Predicted Scanner: 
                            <span style="color:#00796b;">{pred}</span></h3>
                        <p style="background-color:#e8f5e9; padding:8px; border-radius:8px; 
                                  display:inline-block; color:#2e7d32;">
                            Prediction made using <b>{model_choice}</b> Model
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                clf = joblib.load(f"models/{'random_forest' if model_choice == 'Random Forest' else 'svm'}.pkl")
                classes = clf.classes_
                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": prob
                })
                st.markdown("### 🔝 Top Prediction Probability")
                st.dataframe(prob_df.sort_values("Probability", ascending=False).head(10).style.format({"Probability": "{:.4f}"}))

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

elif menu == "🧩 Baseline Feature Extraction":
    st.subheader("🧩 Baseline Feature Extraction Utility")
    dataset_root = st.text_input("Enter dataset root path:", "")

    if dataset_root:
        if os.path.isdir(dataset_root):
            with st.spinner("Extracting features, please wait..."):
                try:
                    df, save_path = baseline_feature_extraction(dataset_root)
                    st.success(f"✅ Features extracted successfully and saved to: {save_path}")
                    st.dataframe(df.head(20))

                    if "class_label" in df.columns:
                        st.subheader("📊 Class Distribution")
                        st.bar_chart(df["class_label"].value_counts())
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.error("Invalid dataset directory path.")
elif menu == "📈 Baseline Model Report":
    st.subheader("📈 Baseline Model Performance Summary")

    metadata_csv = st.text_input("Path to metadata_features.csv", "")
    models_dir = st.text_input("Directory containing baseline models (.joblib/.pkl)", "")

    if st.button("Generate Report"):
        if not metadata_csv or not os.path.exists(metadata_csv):
            st.error("Please provide a valid metadata CSV path.")
        elif not models_dir or not os.path.isdir(models_dir):
            st.error("Please provide a valid models directory.")
        else:
            with st.spinner("Evaluating models..."):
                try:
                    results = evaluate_baseline_models(metadata_csv, models_dir)
                    for res in results:
                        st.markdown(f"### 🧠 Model: **{res['model']}**")
                        if "error" in res:
                            st.error(f"❌ Error: {res['error']}")
                            continue

                        st.write(f"**Accuracy:** {res['accuracy']*100:.2f}%")

                        # --- Confusion Matrix ---
                        cm = res["confusion_matrix"]
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        st.pyplot(fig)

                        # --- Classification Report ---
                        cls_rep = pd.DataFrame(res["report"]).transpose()
                        st.dataframe(cls_rep.style.format(precision=3))

                except Exception as e:
                    st.error(f"Error during evaluation: {e}")

