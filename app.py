import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

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