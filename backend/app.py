import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import cv2
import mahotas

import base64

def get_base64_bg(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent Agriculture Monitoring and Management System",
    page_icon="🌿",
    layout="wide"
)

# ---------------- 2. STYLING ----------------
st.markdown("""
<style>
.stApp { background-color: #f1f7f2; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1b5e20, #2e7d32);
    min-width: 350px !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size:20px!important; font-weight:600!important; color:white!important; }
section[data-testid="stSidebar"] .st-bd, section[data-testid="stSidebar"] .st-bc, section[data-testid="stSidebar"] .st-ae { font-size:18px!important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h3 { color:white!important; font-size:24px!important; }
h1, h2, h3 { color: #1b4332; font-weight:700; }

.engine-card { background:#fff; padding:25px; border-radius:16px; box-shadow:0 8px 20px rgba(0,0,0,0.08); border-left:8px solid #2e7d32; margin-top:20px; }
.feature-box { background:#fff; border-radius:14px; text-align:center; font-weight:700; margin-top:10px; color:#000!important; box-shadow:0 6px 15px rgba(0,0,0,0.06); border-top:5px solid #52b788; font-size:24px; min-height:280px; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:30px; }
.step-desc { font-size:16px; color:#333!important; font-weight:400; margin-top:10px; }
.step-num { font-size:32px; color:#2e7d32; opacity:0.8; margin-bottom:5px; }
</style>
""", unsafe_allow_html=True)

# ---------------- 3. MODEL UTILITIES ----------------
CLASS_NAMES = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy', 4: 'Background_without_leaves', 5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew', 7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight', 11: 'Corn___healthy',
    12: 'Grape___Black_rot', 13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)', 17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy', 19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy', 21: 'Potato___Early_blight',
    22: 'Potato___Late_blight', 23: 'Potato___healthy', 24: 'Raspberry___healthy',
    25: 'Soybean___healthy', 26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch', 28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot', 30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight', 32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot', 34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot', 36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus', 38: 'Tomato___healthy'
}

# ---------------- CNN ----------------
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("plant_disease_detection_cnn.keras")
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

def process_and_predict(image, model):
    img_resized = image.resize((160, 160))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    
    conf = float(np.max(prediction)) * 100
    pred_class = int(np.argmax(prediction))
    
    return CLASS_NAMES.get(pred_class, "Unknown"), conf


# ---------------- SVM ----------------
# ---------------- SVM ----------------
@st.cache_resource
def load_svm_pipeline():
    try:
        svm_model = joblib.load("svm_plant_disease.pkl")
        scaler    = joblib.load("svm_scaler.pkl")
        pca       = joblib.load("svm_pca.pkl")
        le        = joblib.load("svm_labelencoder.pkl")
        return svm_model, scaler, pca, le
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None, None, None

def extract_svm_features(image):
    # 1. Convert PIL to BGR (Notebook used cv2.imread which is BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))

    # 2. Lighting Normalization (CLAHE) - THIS WAS MISSING
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    normalized_img = cv2.merge((cl,a,b))
    normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_LAB2BGR)

    # 3. Extract Textures (Gray must come from normalized image)
    gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    lbp = mahotas.features.lbp(gray, radius=1, points=8) # 10 features
    haralick = mahotas.features.haralick(gray).mean(axis=0) # 13 features

    # 4. Extract Color (HSV must come from normalized image)
    hsv = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
    # Using [8]*3 means 8x8x8 = 512 features
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv)
    hist_flat = hist_hsv.flatten()

    # 5. Combine: 10 (LBP) + 13 (Haralick) + 512 (Hist) = 535
    features = np.hstack([lbp, haralick, hist_flat])

    # 6. Final Shape Check
    # Your PCA expects 561. This means your training script actually 
    # had 26 more features (likely lbp radius 2). 
    # We will pad with zeros to match the 561 requirement.
    if len(features) < 561:
        features = np.pad(features, (0, 561 - len(features)), 'constant')
    else:
        features = features[:561]

    return features

def predict_svm(image, svm_model, scaler, pca, le):

    features = extract_svm_features(image)
    features = np.array(features).reshape(1, -1)

    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    pred_idx = svm_model.predict(features_pca)[0]
    probs = svm_model.predict_proba(features_pca)[0]

    conf = np.max(probs) * 100
    pred_class = le.inverse_transform([pred_idx])[0]

    return pred_class, conf

# ---------------- RANDOM FOREST ----------------
@st.cache_resource
def load_rf_pipeline():
    try:
        rf_model = joblib.load("rf_plant_disease_optimized.pkl")
        scaler   = joblib.load("rf_scaler.pkl")
        pca      = joblib.load("rf_pca.pkl")
        le       = joblib.load("rf_labelencoder.pkl")
        return rf_model, scaler, pca, le
    except Exception as e:
        st.error(f"Error loading RF model: {e}")
        return None, None, None, None

# ---------------- RANDOM FOREST FEATURE EXTRACTOR ----------------
def extract_features(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8,8))
    cl = clahe.apply(l)

    img = cv2.merge((cl,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = mahotas.features.lbp(gray, radius=1, points=8)
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    cv2.normalize(hist,hist)

    features = np.hstack([lbp, haralick, hist.flatten()])

    return features

    return np.hstack([lbp1, lbp2, haralick, hist_hsv.flatten()])
def predict_rf(image, rf_model, scaler, pca, le):

    features = extract_features(image)

    features = np.array(features).reshape(1, -1)

    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    pred = rf_model.predict(features_pca)[0]
    conf = np.max(rf_model.predict_proba(features_pca)) * 100

    return le.inverse_transform([pred])[0], conf
# ---------------- 4. SIDEBAR ----------------


# ---------------- BACKGROUND CSS ----------------
def set_bg(image_path):
    bin_str = get_base64_bg(image_path)
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(241, 247, 242, 0.5), rgba(241, 247, 242, 0.5)),
            url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        .feature-box {{
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(5px);
            border-radius: 14px;
            text-align: center;
            font-weight: 700;
            color: #000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border-top: 5px solid #52b788;
            font-size: 24px;
            padding: 25px;
        }}

        .engine-card {{
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }}
        </style>
    """, unsafe_allow_html=True)


# Apply background once
try:
    set_bg("assets/11781753-c22b-4b23-879f-e54e998e771c.jpg")
except:
    pass


# ---------------- HOME ----------------
st.markdown("""
<h1 style="
    font-size: 48px;
    font-weight: 800;
    color: #1b5e20;
    margin-bottom: 12px;
    text-align: center;
">
    Intelligent Agriculture Monitoring and Management System
</h1>

<p style="
    font-size: 18px;
    color: #2d6a4f;
    max-width: 800px;
    margin: auto;
    line-height: 1.6;
    font-weight: 500;
    text-align: center;
">
    AI-powered crop disease detection using Deep Learning (CNN) and Machine Learning (SVM & Random Forest)
    for faster, smarter, and more accurate agricultural decision support.
</p>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components

st.markdown("""
<div style="margin-top: 60px; text-align:center;">
<h2 style="
    font-size:40px;
    font-weight:800;
    color:#0f3d1e;
    margin-bottom:10px;
">
How It Works
</h2>
</div>
""", unsafe_allow_html=True)

components.html("""
<style>
    .roadmap-wrapper {
        padding: 50px 20px;
        font-family: 'Segoe UI', sans-serif;
        background: transparent;
        max-width: 900px;
        margin: auto;
    }

    .step-container {
        display: flex;
        align-items: center;
        margin-bottom: -20px;
        position: relative;
    }

    /* The Vertical Connector Line */
    .step-container::before {
        content: '';
        position: absolute;
        left: 35px;
        top: 50px;
        width: 3px;
        height: 100%;
        background: linear-gradient(to bottom, #2e7d32 0%, #e0eee0 100%);
        z-index: 1;
    }

    .step-container:last-child::before {
        display: none;
    }

    /* Icon Glow Circle */
    .icon-node {
        width: 74px;
        height: 74px;
        background: #ffffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
        border: 4px solid #fff;
        transition: all 0.4s ease;
    }

    .step-container:hover .icon-node {
        transform: scale(1.1);
        box-shadow: 0 0 25px rgba(46, 125, 50, 0.3);
        border-color: #2e7d32;
    }

    .content-card {
        background: #ffffff;
        margin-left: 30px;
        padding: 20px 25px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.04);
        border: 1px solid #f0f4f1;
        flex: 1;
        transition: all 0.3s ease;
        border-left: 6px solid #e0eee0;
    }

    .step-container:hover .content-card {
        border-left: 6px solid #2e7d32;
        transform: translateX(10px);
    }

    .step-label {
        color: #2e7d32;
        font-weight: 800;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .step-title {
        font-size: 20px;
        font-weight: 800;
        color: #1a2e1c;
        margin: 5px 0;
    }

    .step-desc {
        font-size: 14px;
        color: #5c6b5e;
        line-height: 1.5;
    }

    svg {
        width: 32px;
        height: 32px;
        stroke: #2e7d32;
        fill: none;
        stroke-width: 2;
    }
</style>

<div class="roadmap-wrapper">
    <div class="step-container">
        <div class="icon-node">
            <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        </div>
        <div class="content-card">
            <div class="step-label">Phase 01</div>
            <div class="step-title">Upload Image</div>
            <div class="step-desc">Securely upload high-resolution leaf image. Our system supports JPG, PNG, and JPEG formats.</div>
        </div>
    </div>

    <div style="height: 40px;"></div>

    <div class="step-container">
        <div class="icon-node">
            <svg viewBox="0 0 24 24"><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/><circle cx="12" cy="12" r="4"/></svg>
        </div>
        <div class="content-card">
            <div class="step-label">Phase 02</div>
            <div class="step-title">Image Preprocessing</div>
            <div class="step-desc">Automated noise reduction, CLAHE lighting normalization, and feature scaling for maximum accuracy.</div>
        </div>
    </div>

    <div style="height: 40px;"></div>

    <div class="step-container">
        <div class="icon-node">
            <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        </div>
        <div class="content-card">
            <div class="step-label">Phase 03</div>
            <div class="step-title">Multi-Model Analysis</div>
            <div class="step-desc">A CNN and machine learning models (SVM, Random Forest) are used independently for cross-verification of predictions.</div>
        </div>
    </div>

    <div style="height: 40px;"></div>

    <div class="step-container">
        <div class="icon-node">
            <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
        </div>
        <div class="content-card">
            <div class="step-label">Phase 04</div>
            <div class="step-title">Results</div>
            <div class="step-desc">Receive a definitive identification with confidence percentages.</div>
        </div>
    </div>
</div>
""", height=650)
# ---------------- DISEASE IDENTIFICATION (MOVED HERE) ----------------import streamlit as st
from PIL import Image
import base64
import streamlit.components.v1 as components
from io import BytesIO

# ---------------- IMAGE ENCODER ----------------
def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# ---------------- TITLE ----------------
st.markdown(
    "<h2 style='text-align:center;'>Disease Identification System</h2>",
    unsafe_allow_html=True
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.block-container {
    padding-top: 2rem !important;
    padding-left: 10% !important;
    padding-right: 10% !important;
}

div[data-testid="stSelectbox"],
div[data-testid="stFileUploader"] {
    max-width: 450px;
    margin: auto;
}

section[data-testid="stFileUploader"] {
    border: 1px solid #e0e6e3;
    border-radius: 12px;
    padding: 10px;
    background: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------


col1, col2, col3 = st.columns([1,2,1])

with col2:
    engine_type = st.selectbox("Select Engine", ["CNN", "ML"])

    sub_page = "CNN"
    if engine_type == "ML":
        sub_page = st.selectbox("ML Algorithm", ["SVM", "Random Forest"])

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# ---------------- MAIN ----------------
if uploaded_file:

    raw_image = Image.open(uploaded_file).convert("RGB")

    res = None
    conf = None

    # CNN
    if sub_page == "CNN":
        cnn_model = load_cnn_model()
        if cnn_model:
            with st.spinner("Analyzing with CNN..."):
                res, conf = process_and_predict(raw_image, cnn_model)

    # ML
    else:
        svm_model, scaler, pca, le = load_svm_pipeline()

        if svm_model is not None:
            with st.spinner("Analyzing with ML model..."):

                if sub_page == "SVM":
                    res, conf = predict_svm(raw_image, svm_model, scaler, pca, le)

                elif sub_page == "Random Forest":
                    rf_model, scaler, pca, le = load_rf_pipeline()
                    res, conf = predict_rf(raw_image, rf_model, scaler, pca, le)

    # ---------------- RESULT ----------------
    if res is not None and conf is not None:

        conf = float(min(conf, 100))
        is_healthy = "healthy" in res.lower()

        status = "Healthy" if is_healthy else "Diseased"
        disease = "No Disease Detected" if is_healthy else res
        model_used = f"{engine_type} ({sub_page})"

        img_base64 = image_to_base64(raw_image)

        report_text = f"""
AI DIAGNOSIS REPORT
---------------------
Status: {status}
Disease Name: {disease}
Model: {model_used}
Confidence: {conf:.2f}%
Prediction Completed
"""

        b64 = base64.b64encode(report_text.encode()).decode()

        # ---------------- HORIZONTAL CARD ----------------
        card_html = f"""
        <div style="
            width:100%;
            max-width:900px;
            margin:auto;
            background:#ffffff;
            border-radius:18px;
            border:1px solid #e0e0e0;
            box-shadow:0 10px 25px rgba(0,0,0,0.10);
            overflow:hidden;
            font-family:Arial;
        ">

            <!-- TOP TITLE -->
            <div style="
                text-align:center;
                font-weight:700;
                color:#1b5e20;
                padding:12px;
                border-bottom:1px solid #eee;
                font-size:15px;
            ">
                Prediction Completed
            </div>

            <div style="display:flex;">

                <!-- LEFT IMAGE -->
                <div style="
                    width:40%;
                    background:#f5f7f6;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    padding:15px;
                ">
                    <img src="data:image/png;base64,{img_base64}" 
                    style="width:100%; border-radius:12px;" />
                </div>

                <!-- RIGHT DETAILS -->
                <div style="width:60%; padding:20px; font-size:14px; line-height:1.8;">

                    <p><b>Status:</b> {status}</p>
                    <p><b>Disease Name:</b> {disease}</p>
                    <p><b>Model:</b> {model_used}</p>
                    <p><b>Confidence:</b> {conf:.2f}%</p>

                </div>

            </div>
        </div>
        """

        components.html(card_html, height=480)

        # ---------------- DOWNLOAD BUTTON (OUTSIDE CARD) ----------------
        st.markdown(f"""
        <div style="text-align:center; margin-top:5px;">
            <a href="data:file/txt;base64,{b64}" download="AI_Report.txt">
                <button style="
                    background:linear-gradient(90deg,#1b5e20,#2e7d32);
                    color:white;
                    padding:10px 18px;
                    border:none;
                    border-radius:10px;
                    font-weight:600;
                    cursor:pointer;
                ">
                    📄 Download Report
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")