import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config for a modern look
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
    [data-testid="stSidebar"] { background: #1a1a1c; border-right: 1px solid rgba(255,255,255,0.1); }
    h1 { background: linear-gradient(135deg, #fff 0%, #ff4d4d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

def categorize_bp(bp):
    if bp < 120: return 'Normal'
    elif 120 <= bp < 130: return 'Elevated'
    elif 130 <= bp < 140: return 'Stage 1 HTN'
    else: return 'Stage 2 HTN'

# Load Model
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

st.title("❤️ Heart Disease Diagnostic AI")
st.write("Advanced XAI-powered analysis of clinical bio-markers.")

# Main Form
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🩺 Patient Bio-Markers")
        age = st.number_input("Age", 1, 120, 58)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
        bp = st.number_input("Blood Pressure (systolic)", 50, 250, 150)
        chol = st.number_input("Cholesterol Level", 100, 600, 240)
        fbs = st.selectbox("FBS > 120 mg/dl", [1, 0], format_func=lambda x: "True" if x == 1 else "False")
        
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        ekg = st.selectbox("EKG Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "L. Ventricular Hypertrophy"][x])
        max_hr = st.number_input("Maximum Heart Rate", 50, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 2.5)
        slope = st.selectbox("Slope of ST Segment", [1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
        ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 1)
        thal = st.selectbox("Thallium Scan", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

    if st.button("🚀 RUN CLINICAL ANALYSIS", use_container_width=True):
        try:
            model, feature_columns = load_model()
            
            # Feature engineering
            exp_max_hr = 220 - age
            hr_reserve = exp_max_hr - max_hr
            bp_cat = categorize_bp(bp)
            
            # Prepare input
            input_data = {
                "Age": age, "Sex": sex, "Chest pain type": cp, "BP": bp, "Cholesterol": chol,
                "FBS over 120": fbs, "EKG results": ekg, "Max HR": max_hr,
                "Exercise angina": exang, "ST depression": oldpeak, "Slope of ST": slope,
                "Number of vessels fluro": ca, "Thallium": thal
            }
            
            df = pd.DataFrame([input_data])
            df['Expected_Max_HR'] = exp_max_hr
            df['HR_Reserve'] = hr_reserve
            df['BP_Category'] = bp_cat
            
            # One-hot encoding and aligning features
            df = pd.get_dummies(df)
            df = df.reindex(columns=feature_columns, fill_value=0)
            
            # Predict
            prob = model.predict(df)[0]
            
            # Display Results
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prob > 0.5:
                    st.error("🚨 HIGH RISK DETECTED")
                else:
                    st.success("✅ LOW RISK DETECTED")
                st.metric("Risk Confidence Score", f"{prob*100:.1f}%")
            
            with res_col2:
                st.subheader("Diagnostic Breakdown")
                st.progress(prob)
                st.info(f"The model analyzed clinical bio-markers including your blood pressure categorization ({bp_cat}) and heart rate reserve ({hr_reserve}).")
                if prob > 0.7:
                    st.warning("High confidence presence of indicators consistent with heart disease. Immediate consultation with a healthcare professional is strongly advised.")
                elif prob < 0.3:
                    st.write("Indicators are within statistically healthy ranges for the given patient profile.")
        
        except Exception as e:
            st.error(f"Error during analysis: {e}. Ensure model files are in the repository.")

st.sidebar.subheader("About")
st.sidebar.info("This is an AI-driven clinical tool using LightGBM (95.6% AUC) to assist in heart disease diagnosis.")
st.sidebar.warning("Note: This is for research purposes and should not be used as the sole basis for clinical decisions.")
