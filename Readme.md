#  AI TraceFinder — Forensic Scanner Identification  

##  Overview  
AI TraceFinder is a forensic machine learning platform that identifies the **source scanner device** used to digitize a document or image. Each scanner (brand/model) introduces unique **noise, texture, and compression artifacts** that serve as a fingerprint. By analyzing these patterns, AI TraceFinder enables **fraud detection, authentication, and forensic validation** in scanned documents.  
# 📊 AI TraceFinder: Forensic Scanner Identification

Detecting document forgery by analyzing a scanner's unique digital fingerprint.

---

## 🎯 About the Project

Scanned documents like legal agreements, official certificates, and financial records are easy to forge. It's often impossible to tell if a scanned document is legitimate or if it was created using an unauthorized, fraudulent device.

*AI TraceFinder* solves this problem by identifying the *source scanner* used to create a digital image.

Every scanner, due to its unique hardware, introduces microscopic and invisible *“fingerprints”* into an image. These include specific noise patterns, texture artifacts, and compression traces.  
This project uses *machine learning and deep learning* to recognize these unique signatures, allowing you to:

- Attribute a scanned document to a specific scanner model.  
- Detect forgeries where unauthorized scanners were used.  
- Verify the authenticity of scanned evidence in a forensic context.

---

## 🧠 Tech Stack

This project leverages a modern stack for machine learning, image processing, and web app delivery.

| Category | Technology | Purpose |
|-----------|-------------|----------|
| *Backend & ML* | Python | Core programming language |
| | Scikit-learn | Random Forest & SVM (Baseline Models) |
| | Pandas | Data manipulation and CSV handling |
| | OpenCV | Image processing (color, noise, filters) |
| | NumPy | Numerical operations |
| | TensorFlow / Keras | CNN model for deep learning |
| *Frontend & UI* | Streamlit | Interactive web application |
| | Matplotlib & Seaborn | Data visualization (plots, confusion matrix) |
| | Pillow (PIL) | Displaying and loading images |
| *Tooling* | Git & GitHub | Version control |
| | venv | Virtual environment management |

---

## ✨ Features

- 🧩 *Feature Extraction Module:* Scan image directories, extract 10+ metadata features, and generate a feature CSV.  
- 📊 *Data Visualization:* Display class distributions, sample images, and preview dataset.  
- 💾 *Downloadable Results:* Export generated feature CSV files directly from the app.  
- 🤖 *Machine Learning Pipeline:*  
  - *Train:* Build and train Random Forest & SVM models.  
  - *Evaluate:* View detailed classification reports and confusion matrices.  
  - *Predict:* Upload any image to identify its source scanner.  
- ⚙ *Dual Model Support:* Choose between Random Forest or SVM for prediction.  
- 🧠 *Deep Learning Integration:* CNN model for end-to-end image-based classification.  
- 🌐 *User-Friendly UI:* Deployed via Streamlit with an intuitive, modular interface.

---
💡 Usage

1. Launch the Streamlit interface.


2. Navigate through the tabs:

Feature Extraction: Upload or select dataset folder.

Visualization: View dataset distribution and sample images.

Model Training: Train Random Forest, SVM, or CNN models.

Prediction: Upload a new image to identify its scanner.



3. View accuracy metrics, download results, or visualize confusion matrices.

## 🖼 Demo / Screenshots

Upload your screenshots here once ready!  
Example structure:

1. *Main Prediction App*
   ![Main Prediction App]<img width="1749" height="1014" alt="Screenshot 2025-11-03 164710" src="https://github.com/user-attachments/assets/6c8eb1ed-ddb7-4bab-a624-3deec9337ef3" />

3. *Feature Extraction Interface*
   ![Feature Extraction Interface](./img/Feature_Extraction_App.png)

4. *Model Evaluation Page*
   ![Model Evaluation]![WhatsApp Image 2025-11-03 at 17 32 54_33340dfe](https://github.com/user-attachments/assets/ca6ea421-e141-4073-b488-fb35dbb9e5c3)


5. *Data Visualization*
   ![Data Visualization]<img width="1919" height="1072" alt="Screenshot 2025-11-03 170622" src="https://github.com/user-attachments/assets/fd0bc96f-67d5-4e44-a262-1f105c8a3987" /><img width="1919" height="1079" alt="Screenshot 2025-11-03 170642" src="https://github.com/user-attachments/assets/f6660b0d-fb83-4cb0-87f8-db61d86bd941" /><img width="1919" height="1079" alt="Screenshot 2025-11-03 170703" src="https://github.com/user-attachments/assets/96d7d50d-e4df-46c8-a6c7-1d4f77bb1ffe" />




---

## 📈 Accuracy & Performance

| Metric | Value |
|--------|--------|
| *Test Accuracy (CNN)* | *82.21%* |
| *Precision* | 0.83 |
| *Recall* | 0.82 |
| *F1-Score* | 0.82 |
| *Test Samples* | 517 images |

---

## 🚀 Installation

Follow these steps to set up the project locally.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/debdotta/AI_TraceFinder.git
cd AI_TraceFinder
---

##  Goals & Objectives  
- Collect and label scanned document datasets from multiple scanners  
- Robust preprocessing (resize, grayscale, normalize, denoise)  
- Extract scanner-specific features (noise, FFT, PRNU, texture descriptors)  
- Train classification models (ML + CNN)  
- Apply explainability tools (Grad-CAM, SHAP)  
- **Deploy an interactive app for scanner source identification**  
- Deliver **accurate, interpretable results** for forensic and legal use cases  

---

##  Methodology 
1. **Data Collection & Labeling**  
   - Gather scans from 3–5 scanner models/brands  
   - Create a structured, labeled dataset  

2. **Preprocessing**  
   - Resize, grayscale, normalize  
   - Optional: denoise to highlight artifacts  

3. **Feature Extraction**  
   - PRNU patterns, FFT, texture descriptors (LBP, edge features)  

4. **Model Training**  
   - Baseline ML: SVM, Random Forest, Logistic Regression  
   - Deep Learning: CNN with augmentation  

5. **Evaluation & Explainability**  
   - Metrics: Accuracy, F1-score, Confusion Matrix  
   - Interpretability: Grad-CAM, SHAP feature maps  

6. **Deployment**  
   - Streamlit app → upload scanned image → predict scanner model  
   - Display confidence score and key feature regions  

---

##  Actionable Insights for Forensics  
- **Source Attribution:** Identify which scanner created a scanned copy of a document.  
- **Fraud Detection:** Detect forgeries where unauthorized scanners were used.  
- **Legal Verification:** Validate whether scanned evidence originated from approved devices.  
- **Tamper Resistance:** Differentiate between authentic vs. tampered scans.  
- **Explainability:** Provide visual evidence of how classification was made.  

---

##  Architecture (Conceptual)  
Input ➜ Preprocessing ➜ Feature Extraction + Modeling ➜ Evaluation & Explainability ➜ Prediction App  

---

## ⏳ 8-Week Roadmap (Milestones)  
- **W1:** Dataset collection (min. 3–5 scanners), labeling, metadata analysis  
- **W2:** Preprocessing pipeline (resize, grayscale, normalize, optional denoise)  
- **W3:** Feature extraction (noise maps, FFT, LBP, texture descriptors)  
- **W4:** Baseline ML models (SVM, RF, Logistic Regression) + evaluation  
- **W5:** CNN model training with augmentation, hyperparameter tuning  
- **W6:** Model evaluation (accuracy, F1, confusion matrix) + Grad-CAM/SHAP analysis  
- **W7:** Streamlit app development → image upload, prediction, confidence output  
- **W8:** Final documentation, results, presentation, and demo handover  

---

##  Suggested Project Structure  
```bash
ai-tracefinder/
├─ app.py              
├─ src/
│  ├─ ingest/           
│  ├─ preprocess/        
│  ├─ features/          
│  ├─ models/            
│  ├─ explain/           
│  └─ utils/             
├─ data/                 
├─ notebooks/            
├─ reports/              
└─ README.md
```
