import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit as st
import pandas as pd
import joblib 

# Deserializing the pickle files
model = joblib.load(os.path.join(BASE_DIR, "../pickle/LogisticRegression_Heart.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../pickle/Scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "../pickle/Columns.pkl"))

st.title("Heart Stroke Prediction")
st.markdown("Enter the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Choleterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0,1])
resting_ecg = st.selectbox("Resting ECG", ['Normal','ST','LVH'])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina",['Y','N'])
old_peak = st.slider("Old Peak (ST Depression)", 0.0, 0.6, 0.1)
st_slope = st.selectbox("ST Slope",['Up','Flat','Down'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': old_peak,
        'Sex_'+sex: 1,
        'ChestPainType_'+chest_pain: 1,
        'RestingECG_'+resting_ecg: 1,
        'ExerciseAngina_'+exercise_angina: 1,
        'ST_Slope_'+st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaling_col = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[scaling_col] = scaler.transform(input_df[scaling_col])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease")
    
        st.markdown("### What you can do now:")
        st.write("• Avoid fried and salty foods, quit smoking if you do, and start light daily exercise after consulting a doctor.")
        st.write("• Check your blood pressure, sugar, and cholesterol regularly and see a heart specialist soon.")
    
    else:
        st.success("✅ Low risk of Heart Disease")
    
        st.markdown("### How to stay healthy:")
        st.write("• Keep your heart healthy by walking at least 30 minutes daily and eating less oily food.")
        st.write("• Limit sugar and salt in meals and get your blood pressure checked once a year.")


