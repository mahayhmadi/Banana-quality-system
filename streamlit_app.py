# -*- coding: utf-8 -*-
import streamlit as st
import sqlite3, hashlib, cv2, os, numpy as np, pandas as pd, tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.saving import register_keras_serializable
from common_config import DATA_ROOT, SQLITE_DB, MODEL_KERAS, CLASS_COLS, IMAGE_SIZE


# =====================================================
# Model setup
# =====================================================
@register_keras_serializable(package="Custom")
def banana_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

@register_keras_serializable(package="Custom")
def weighted_categorical_crossentropy(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

custom_objects = {
    "banana_loss": banana_loss,
    "weighted_categorical_crossentropy": weighted_categorical_crossentropy
}

@st.cache_resource
def load_keras_model(path):
    return load_model(path, safe_mode=False, custom_objects=custom_objects)

model = load_keras_model(MODEL_KERAS)


# =====================================================
# Helpers
# =====================================================
def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()

def preprocess_img_bytes(file_bytes):
    npimg = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE)
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return rgb, resized, arr


# =====================================================
# DB Schema
# =====================================================
def ensure_schema():
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS Produce_Samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            item_name TEXT,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        );
        CREATE TABLE IF NOT EXISTS Quality_Results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            quality_class TEXT,
            confidence REAL,
            freshness_index REAL,
            FOREIGN KEY(sample_id) REFERENCES Produce_Samples(sample_id)
        );
        CREATE TABLE IF NOT EXISTS Shelf_Life_Metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            predicted_storage_days INTEGER,
            optimal_temp_C REAL,
            mock_decay_rate REAL,
            FOREIGN KEY(sample_id) REFERENCES Produce_Samples(sample_id)
        );
    """)
    conn.commit(); conn.close()

ensure_schema()


# =====================================================
# UI CONFIG
# =====================================================
st.set_page_config(page_title="BananaScan", layout="wide")

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] {
    background-color: black !important;
    color: #FFD700 !important;
}
[data-testid="stHeader"] { background: none !important; }
h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; }
p, label, span { color: white !important; }
hr { border: none; height: 1px; background: linear-gradient(to right, #FFD700, transparent); }

/* إخفاء المربع الرمادي */
[data-testid="stFileUploaderDropzone"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] div {
    display: none !important;
}

/* زر التصفح */
button[kind="secondary"] {
    background-color: black !important;
    color: #FFD700 !important;
    border: 1.5px solid #FFD700 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 7px 20px !important;
    font-size: 14px !important;
    transition: all 0.25s ease-in-out;
    display: block !important;
    margin: 15px auto 0 auto !important;
}
button[kind="secondary"]:hover {
    background-color: #FFD700 !important;
    color: black !important;
    transform: scale(1.05);
    box-shadow: 0 0 10px rgba(255,215,0,0.4);
}

/* أيقونات */
.icon {
    width: 48px;
    height: 48px;
    stroke: #FFD700;
    stroke-width: 2;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 6px rgba(255,215,0,0.5));
}

/* بطاقة رفع الملفات */
.upload-card {
    background: #1A1A1A;
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 14px;
    padding: 30px 15px 35px 15px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-card:hover {
    background: #242424;
    box-shadow: 0 0 15px rgba(255,215,0,0.25);
}
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #FFD700;
    margin-bottom: 10px;
}

/* تنسيق النتائج */
.metric-grid {
    display: flex; justify-content: space-between; gap: 15px; margin-top: 25px;
}
.metric-card {
    flex: 1; background: #1A1A1A; border: 1px solid rgba(255,215,0,0.25);
    border-radius: 12px; padding: 18px 10px; text-align: center;
    box-shadow: 0 0 10px rgba(255,215,0,0.08);
    transition: all .3s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 15px rgba(255,215,0,0.25);
}
.metric-title {
    color: #FFD700; font-weight: 700; font-size: 16px; margin-bottom: 5px;
}
.metric-value {
    color: white; font-size: 20px; font-weight: 600;
}

/* تصغير الصورة */
img {
    border-radius: 12px;
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 40% !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# Layout
# =====================================================
col1, col2 = st.columns([1.2, 2.8], gap="large")

with col1:
    st.markdown("""
        <div style="position: fixed; top:0; left:0; width:33%; height:100vh; overflow:hidden;">
            <iframe width="100%" height="100%" 
                src="https://www.youtube.com/embed/7Oi89KlXjYQ?autoplay=1&mute=1&loop=1&playlist=7Oi89KlXjYQ"
                frameborder="0" allow="autoplay; loop; muted"></iframe>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h1>🍌 BananaScan: AI-Powered Banana Quality & Shelf-Life Predictor</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='
        font-size:17px; color:white; font-style:italic;
        line-height:1.6; margin-top:10px; margin-bottom:25px;
        text-align:justify; background:rgba(255,215,0,0.05);
        padding:15px 20px; border-left:3px solid #FFD700;
        border-radius:6px; box-shadow:0 0 6px rgba(255,215,0,0.1);
    '>
        <p><em>Stop guessing and start predicting.</em></p>
        <p><em>BananaScan uses AI to instantly assess banana quality, ripeness stage, and shelf life.</em></p>
        <p><em>Maximize profit, minimize waste.</em></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Upload Section
    st.markdown("<h4 class='section-title'>📤 Upload your banana data</h4>", unsafe_allow_html=True)
    col_img, col_vid = st.columns(2)

    with col_img:
        st.markdown("""
        <div class='upload-card'>
            <svg class='icon' viewBox="0 0 24 24" fill="none">
              <rect x="3" y="7" width="18" height="14" rx="2" ry="2"></rect>
              <circle cx="12" cy="14" r="4"></circle>
            </svg>
            <p><b>Upload Image</b></p>
            <small>Drag and drop or click below<br>Limit 200MB • JPG, JPEG, PNG</small>
        </div>
        """, unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img")

    with col_vid:
        st.markdown("""
        <div class='upload-card'>
            <svg class='icon' viewBox="0 0 24 24" fill="none">
              <rect x="3" y="5" width="15" height="14" rx="2" ry="2"></rect>
              <polygon points="19,7 23,10 23,14 19,17"></polygon>
            </svg>
            <p><b>Upload Video</b></p>
            <small>Drag and drop or click below<br>Limit 200MB • MP4, MOV, AVI</small>
        </div>
        """, unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], label_visibility="collapsed", key="vid")

    # Prediction Section
    if uploaded_image:
        bytes_data = uploaded_image.read()
        orig, _, arr = preprocess_img_bytes(bytes_data)

        class_probs, fresh_val = model.predict(arr, verbose=0)
        idx = int(np.argmax(class_probs[0]))
        cls_name = CLASS_COLS[idx]
        fresh = float(fresh_val[0][0])
        shelf_days = int(max(1, round(fresh / 10 * 7)))

        # الصورة
        st.image(orig, caption="Uploaded Image", use_container_width=False)

        # ✅ المربعات الثلاثة
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-title">🍌 Class</div>
            <div class="metric-value">{cls_name.upper()}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">🌿 Freshness Index</div>
            <div class="metric-value">{fresh:.2f}/10</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">⏳ Shelf Life</div>
            <div class="metric-value">{shelf_days} days</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ✅ Pie Chart وسط الصفحة تماماً
        conn = sqlite3.connect(SQLITE_DB)
        df = pd.read_sql("SELECT quality_class, COUNT(*) AS count FROM Quality_Results GROUP BY quality_class", conn)
        conn.close()

        if not df.empty:
            colors = {
                "unripe": "#32CD32",
                "freshunripe": "#ADFF2F",
                "freshripe": "#FFD700",
                "ripe": "#FFF176",
                "overripe": "#B8860B",
                "rotten": "#3E2723"
            }
            fig = px.pie(df, names="quality_class", values="count", color="quality_class", color_discrete_map=colors)
            fig.update_layout(
                paper_bgcolor="black",
                plot_bgcolor="black",
                height=480,  # حجم أكبر
                width=480,
                font=dict(color="#FFD700", size=13),
                margin=dict(t=60, b=40, l=40, r=40),
                title=dict(text="🍌 Class Distribution", font=dict(color="#FFD700", size=18), y=0.97),
                legend=dict(orientation="v", x=1.05, y=0.5, font=dict(size=11, color="#FFD700"))
            )

            # 👇 مركز الرسم تماماً داخل الحاوية السوداء
            centered_html = """
            <div style='display:flex; justify-content:center; align-items:center;'>
            """
            st.markdown(centered_html, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("No data available yet for pie chart.")

    elif uploaded_video:
        st.video(uploaded_video)
        st.success("🎥 Video uploaded successfully!")
