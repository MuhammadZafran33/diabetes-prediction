# ✅ LINE 1-9: Imports FIRST
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ LINE 10: set_page_config SECOND (must be first Streamlit call)
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# ✅ LINE 15+: CSS THIRD
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════
   GLOBAL RESET & BASE
═══════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1628 100%) !important;
    min-height: 100vh;
}

/* ═══════════════════════════════════════
   ANIMATED BACKGROUND PARTICLES
═══════════════════════════════════════ */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: 
        radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(99,102,241,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(16,185,129,0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ═══════════════════════════════════════
   HEADER / MAIN TITLE
═══════════════════════════════════════ */
h1 {
    background: linear-gradient(135deg, #60a5fa, #818cf8, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    text-align: center !important;
    letter-spacing: -1px !important;
    line-height: 1.2 !important;
    animation: shimmer 3s ease-in-out infinite alternate;
    margin-bottom: 0.2rem !important;
}

@keyframes shimmer {
    0%   { filter: brightness(1) drop-shadow(0 0 20px rgba(96,165,250,0.3)); }
    100% { filter: brightness(1.2) drop-shadow(0 0 35px rgba(129,140,248,0.5)); }
}

h2 {
    color: #93c5fd !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    letter-spacing: -0.3px !important;
}

h3 {
    color: #7dd3fc !important;
    font-weight: 600 !important;
}

/* ═══════════════════════════════════════
   SIDEBAR — PREMIUM GLASSMORPHISM
═══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #93c5fd !important;
    -webkit-text-fill-color: #93c5fd !important;
    font-size: 1.1rem !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #94a3b8 !important;
}

/* Slider Labels */
[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

/* Slider Track */
[data-testid="stSlider"] [data-baseweb="slider"] {
    padding: 8px 0 !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: 3px solid #1e293b !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.6) !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:hover {
    transform: scale(1.2) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.9) !important;
}

/* Slider value */
[data-testid="stSlider"] p {
    color: #60a5fa !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* ═══════════════════════════════════════
   PREDICT BUTTON — PREMIUM GLOW
═══════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 24px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    box-shadow: 0 4px 25px rgba(99,102,241,0.5), 0 0 0 0 rgba(99,102,241,0.4) !important;
    transition: all 0.3s ease !important;
    animation: pulse-btn 2.5s infinite !important;
    margin-top: 8px !important;
}

@keyframes pulse-btn {
    0%   { box-shadow: 0 4px 25px rgba(99,102,241,0.5), 0 0 0 0 rgba(99,102,241,0.4); }
    70%  { box-shadow: 0 4px 25px rgba(99,102,241,0.5), 0 0 0 12px rgba(99,102,241,0); }
    100% { box-shadow: 0 4px 25px rgba(99,102,241,0.5), 0 0 0 0 rgba(99,102,241,0); }
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 35px rgba(99,102,241,0.7) !important;
}

/* ═══════════════════════════════════════
   METRIC CARDS — GLOWING GLASS
═══════════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9)) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
}

[data-testid="stMetric"]:hover {
    border-color: rgba(99,102,241,0.5) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.2) !important;
}

[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* ═══════════════════════════════════════
   INFO / SUCCESS / ERROR BOXES
═══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
    font-weight: 500 !important;
}

/* High Risk Alert */
.stAlert[data-type="error"] {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(153,27,27,0.1)) !important;
    border-left: 4px solid #ef4444 !important;
    box-shadow: 0 4px 20px rgba(239,68,68,0.15) !important;
}

/* Low Risk Alert */
.stAlert[data-type="success"] {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1)) !important;
    border-left: 4px solid #10b981 !important;
    box-shadow: 0 4px 20px rgba(16,185,129,0.15) !important;
}

/* ═══════════════════════════════════════
   DATAFRAME / TABLE
═══════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

[data-testid="stDataFrame"] table {
    background: rgba(15,23,42,0.8) !important;
}

[data-testid="stDataFrame"] th {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    color: #60a5fa !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(99,102,241,0.2) !important;
}

[data-testid="stDataFrame"] td {
    color: #cbd5e1 !important;
    padding: 10px 16px !important;
    border-bottom: 1px solid rgba(30,41,59,0.5) !important;
    font-size: 0.9rem !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: rgba(99,102,241,0.08) !important;
}

/* ═══════════════════════════════════════
   DIVIDERS
═══════════════════════════════════════ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent) !important;
    margin: 2rem 0 !important;
}

/* ═══════════════════════════════════════
   GENERAL MARKDOWN TEXT
═══════════════════════════════════════ */
p, .stMarkdown p {
    color: #94a3b8 !important;
    line-height: 1.7 !important;
}

/* ═══════════════════════════════════════
   COLUMN CONTAINERS — CARD LOOK
═══════════════════════════════════════ */
[data-testid="column"] > div > div > div > [data-testid="stVerticalBlock"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(15,23,42,0.8)) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    padding: 16px !important;
    backdrop-filter: blur(8px) !important;
}

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 3px;
}

/* ═══════════════════════════════════════
   SPINNER
═══════════════════════════════════════ */
[data-testid="stSpinner"] {
    color: #60a5fa !important;
}

/* ═══════════════════════════════════════
   PLOTLY / MATPLOTLIB CHARTS
═══════════════════════════════════════ */
[data-testid="stPlotlyChart"],
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}

/* ═══════════════════════════════════════
   TOOLTIP / HELP ICONS
═══════════════════════════════════════ */
[data-testid="stTooltipIcon"] {
    color: #4f46e5 !important;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
footer { display: none !important; }
#MainMenu { display: none !important; }
header { visibility: hidden !important; }

/* ═══════════════════════════════════════
   ENTRY ANIMATION
═══════════════════════════════════════ */
[data-testid="stVerticalBlock"] {
    animation: fadeInUp 0.5s ease forwards;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════
   SUBTITLE / CAPTION TEXT
═══════════════════════════════════════ */
[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-size: 0.8rem !important;
    text-align: center !important;
}

/* Subtitle banner */
.subtitle-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.1));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 10px 20px;
    text-align: center;
    color: #64748b !important;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    letter-spacing: 0.5px;
}

/* Risk badge */
.risk-high {
    display: inline-block;
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(239,68,68,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 8px;
    padding: 4px 12px;
    color: #f87171;
    font-weight: 700;
    font-size: 0.85rem;
}

.risk-low {
    display: inline-block;
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(52,211,153,0.1));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 8px;
    padding: 4px 12px;
    color: #34d399;
    font-weight: 700;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ✅ REST of your app code below...

# Load model files
@st.cache_resource
def load_model():
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-positive {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 20px;
        border-radius: 5px;
        font-size: 18px;
    }
    .result-negative {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 5px;
        font-size: 18px;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏥 Diabetes Risk Predictor</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-Powered Early Detection System | Random Forest Model | 74% Accuracy</p>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Patient Information")
st.sidebar.markdown("Enter patient details below:")

# Input fields in sidebar
pregnancies = st.sidebar.slider(
    "Pregnancies", 0, 17, 3,
    help="Number of times pregnant"
)
glucose = st.sidebar.slider(
    "Glucose Level (mg/dL)", 50, 200, 120,
    help="Plasma glucose concentration"
)
blood_pressure = st.sidebar.slider(
    "Blood Pressure (mm Hg)", 20, 130, 70,
    help="Diastolic blood pressure"
)
skin_thickness = st.sidebar.slider(
    "Skin Thickness (mm)", 5, 100, 25,
    help="Triceps skin fold thickness"
)
insulin = st.sidebar.slider(
    "Insulin Level (IU/mL)", 10, 850, 80,
    help="2-Hour serum insulin"
)
bmi = st.sidebar.slider(
    "BMI", 10.0, 70.0, 25.0, 0.1,
    help="Body Mass Index"
)
dpf = st.sidebar.slider(
    "Diabetes Pedigree Function", 0.05, 2.5, 0.5, 0.01,
    help="Diabetes hereditary risk score"
)
age = st.sidebar.slider(
    "Age (years)", 18, 90, 33,
    help="Patient age"
)

# Predict button
predict_btn = st.sidebar.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Model", "Random Forest")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Accuracy", "74.03%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Dataset", "768 patients")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Prediction section
if predict_btn:
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    col_result, col_chart = st.columns([1, 1])
    
    with col_result:
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                ⚠️ <strong>HIGH DIABETES RISK DETECTED</strong><br><br>
                Risk Probability: <strong>{probability[1]*100:.1f}%</strong><br>
                Confidence: <strong>{max(probability)*100:.1f}%</strong><br><br>
                <em>Please consult a healthcare professional immediately.</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                ✅ <strong>LOW DIABETES RISK</strong><br><br>
                Risk Probability: <strong>{probability[1]*100:.1f}%</strong><br>
                Confidence: <strong>{max(probability)*100:.1f}%</strong><br><br>
                <em>Maintain healthy lifestyle to keep risk low.</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Input summary
        st.subheader("📊 Patient Summary")
        summary_df = pd.DataFrame({
            'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure',
                         'Skin Thickness', 'Insulin', 'BMI',
                         'Pedigree Function', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi, dpf, age]
        })
        st.dataframe(summary_df, use_container_width=True)
    
    with col_chart:
        st.subheader("📈 Risk Analysis")
        
        # Probability gauge chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Pie chart - risk probability
        sizes = [probability[0]*100, probability[1]*100]
        colors = ['#4CAF50', '#F44336']
        explode = (0, 0.1)
        ax1.pie(sizes, explode=explode, labels=['No Diabetes', 'Diabetes'],
                colors=colors, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12})
        ax1.set_title('Risk Distribution', fontweight='bold')
        
        # Feature importance bar chart
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure',
                        'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance')
        
        ax2.barh(feat_df['Feature'], feat_df['Importance'],
                color='#2196F3', edgecolor='black', linewidth=0.5)
        ax2.set_title('Feature Importance', fontweight='bold')
        ax2.set_xlabel('Importance Score')
        
        plt.tight_layout()
        st.pyplot(fig)

else:
    # Show welcome message
    st.info("👈 Enter patient details in the sidebar and click 'Predict Diabetes Risk' to get results!")
    
    st.subheader("ℹ️ About This App")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **How it works:**
        - Enter patient medical data in the sidebar
        - Our Random Forest model analyzes the data
        - Get instant diabetes risk prediction
        - See probability score and feature analysis
        """)
    
    with col_b:
        st.markdown("""
        **Features analyzed:**
        - Glucose levels
        - BMI and Age
        - Blood Pressure
        - Insulin levels
        - Family history (Pedigree)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 14px;'>
    ⚠️ Disclaimer: This tool is for educational purposes only. 
    Always consult a qualified healthcare professional for medical advice.
    <br>Built by Zafii ML | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)