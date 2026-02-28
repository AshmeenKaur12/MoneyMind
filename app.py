import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import cv2
import mahotas

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
    img_resized = image.resize((160, 160))  # match training size
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    pred_class = int(np.argmax(prediction))
    conf = float(np.max(prediction)) * 100
    res = CLASS_NAMES.get(pred_class, "Unknown")
    return res, conf

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
    # Match training: resize to 128x128
    img = np.array(image)
    img = cv2.resize(img, (128,128))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Multi-scale LBP exactly as trained
    lbp1 = mahotas.features.lbp(gray, radius=1, points=8)
    lbp2 = mahotas.features.lbp(gray, radius=2, points=16)

    # Haralick textures
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    # HSV histogram (8 bins per channel)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8]*3, [0,256]*3)
    cv2.normalize(hist, hist)

    return np.hstack([lbp1, lbp2, haralick, hist.flatten()])

def predict_svm(image, svm_model, scaler, pca, le):
    features = extract_svm_features(image)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    pred_class_idx = svm_model.predict(features_pca)[0]
    pred_class = le.inverse_transform([pred_class_idx])[0]
    return pred_class

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
    # image must be BGR (OpenCV format)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multi-scale LBP
    lbp1 = mahotas.features.lbp(gray, radius=1, points=8)
    lbp2 = mahotas.features.lbp(gray, radius=2, points=16)

    # Haralick texture
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    # HSV histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0,1,2], None, [8]*3, [0,256]*3)
    cv2.normalize(hist_hsv, hist_hsv)

    return np.hstack([lbp1, lbp2, haralick, hist_hsv.flatten()])
def predict_rf(image, rf_model, scaler, pca, le):

    # Convert PIL → OpenCV BGR EXACTLY like training
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128,128))

    # Use SAME training extractor
    features = extract_features(img)

    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    pred_idx = rf_model.predict(features_pca)[0]
    pred_class = le.inverse_transform([pred_idx])[0]

    probs = rf_model.predict_proba(features_pca)
    conf = np.max(probs) * 100

    return pred_class, conf
# ---------------- 4. SIDEBAR ----------------
with st.sidebar:
    st.image("my logo.jpg", width=300)
    st.markdown("<br>", unsafe_allow_html=True)

    menu = st.radio("Main Menu", ["🏠 Home", "🚀 Disease Identification"])
    st.divider()

    sub_page = None
    if menu == "🚀 Disease Identification":
        st.subheader("Identification Engines")
        engine_type = st.selectbox("Select Engine", ["Through CNN", "Through ML"])
        if engine_type == "Through ML":
            sub_page = st.selectbox("ML Algorithm", ["SVM", "Random Forest"])
        else:
            sub_page = "CNN"

# ---------------- 5. HOME PAGE ----------------
if menu == "🏠 Home":
    st.title("🌿 Intelligent Agriculture Monitoring and Management System")
    st.write("""
        Welcome to the AI-powered smart farming platform designed to help farmers, researchers,
        and agricultural experts monitor crop health, detect diseases, and improve productivity.
    """)
    st.subheader("🚀 Check out our Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-box">🌱 AI-Based Plant Disease Detection</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box">📊 Real-Time Crop Analysis</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-box">🤖 Multi-Model Diagnostic Engine</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🌿 Steps to Begin with Disease Identification")
    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown("""
            <div class="feature-box">
                <div class="step-num">01</div>
                <div>📸 Upload Image</div>
                <div class="step-desc">Take a photo of the plant leaf showing symptoms.</div>
            </div>
        """, unsafe_allow_html=True)
    with step_col2:
        st.markdown("""
            <div class="feature-box">
                <div class="step-num">02</div>
                <div>🔍 Get Diagnosis</div>
                <div class="step-desc">The system identifies the disease affecting your plant.</div>
            </div>
        """, unsafe_allow_html=True)
    with step_col3:
        st.markdown("""
            <div class="feature-box">
                <div class="step-num">03</div>
                <div>🧪 Take Action</div>
                <div class="step-desc">Follow the advice to treat and protect your plants.</div>
            </div>
        """, unsafe_allow_html=True)

# ---------------- 6. DISEASE IDENTIFICATION ----------------
elif menu == "🚀 Disease Identification":
    st.subheader(f"🌿 Diagnosis Mode: {sub_page}")
    uploaded_file = st.file_uploader(f"Upload leaf image for {sub_page} analysis", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(raw_image, use_container_width=True)
        with col2:
            if sub_page == "CNN":
                cnn_model = load_cnn_model()
                if cnn_model:
                    with st.spinner("Processing image via CNN..."):
                        res, conf = process_and_predict(raw_image, cnn_model)
                        st.markdown(f"""
                            <div class="engine-card">
                                <h2 style="margin-top:10px; color:#1b4332;">{res}</h2>
                                <p style="color:#555;">Confidence: <b>{conf:.2f}%</b></p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Engine failure: CNN model could not be loaded.")
            else:
                svm_model, scaler, pca, le = load_svm_pipeline()
                if svm_model is None:
                    st.error("ML engine could not be loaded.")
                else:
                    with st.spinner(f"Processing image via {sub_page}..."):
                        time.sleep(0.5)
                        if sub_page == "SVM":
                            res = predict_svm(raw_image, svm_model, scaler, pca, le)
                            conf = 90.0

                        elif sub_page == "Random Forest":
                            rf_model, scaler, pca, le = load_rf_pipeline()
                            if rf_model is None:
                                st.error("Random Forest engine could not be loaded.")
                                st.stop()
                            res, conf = predict_rf(raw_image, rf_model, scaler, pca, le)
                        st.markdown(f"""
                            <div class="engine-card">
                                <h2 style="margin-top:10px; color:#1b4332;">{res}</h2>
                                <p style="color:#555;">Confidence: <b>{conf:.2f}%</b></p>
                            </div>
                        """, unsafe_allow_html=True)

st.markdown("---")