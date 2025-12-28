import streamlit as st
import pandas as pd
import joblib

# Load pipeline and label encoder
pipeline = joblib.load("obesity_pipeline.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.title("Obesity Level Prediction App")

st.write("Enter your details below to predict obesity level:")

# Collect user inputs (all 16 features)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
height = st.number_input("Height (m)", min_value=1.0, max_value=3.0, value=1.70)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
favc = st.selectbox("Do you eat high caloric food frequently (FAVC)?", ["yes", "no"])
fcvc = st.number_input("Frequency of vegetable consumption per day (FCVC)", min_value=1, max_value=3, value=2)
ncp = st.number_input("Number of main meals (NCP)", min_value=1, max_value=4, value=3)
caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
CH2O = st.number_input("Daily water consumption (CH2O, liters)", min_value=0.5, max_value=3.0, value=2.0)
scc = st.selectbox("Do you monitor your calories (SCC)?", ["yes", "no"])
faf = st.number_input("Physical activity frequency (FAF, hours per week)", min_value=0.0, max_value=100.0, value=2.0)
tue = st.number_input("Time using technology devices (TUE, hours per day)", min_value=0.0, max_value=24.0, value=2.0)
calc = st.selectbox("Consumption of alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportation used (MTRANS)", ["Bike", "Walking", "Motorbike", "Automobile", "Public_Transportation"])

# Build input dataframe (raw, pipeline will preprocess)
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "family_history_with_overweight": [family_history],
    "FAVC": [favc],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC": [caec],
    "SMOKE": [smoke],
    "CH2O": [CH2O],
    "SCC": [scc],
    "FAF": [faf],
    "TUE": [tue],
    "CALC": [calc],
    "MTRANS": [mtrans]
})

# Predict
if st.button("Predict"):
    prediction = pipeline.predict(input_data)
    decoded_prediction = label_encoder.inverse_transform(prediction)

    proba = pipeline.predict_proba(input_data)[0]
    proba_df = pd.DataFrame({
        "Obesity Level": label_encoder.classes_,
        "Probability (%)": (proba * 100).round(2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.success(f"Predicted Obesity Level: {decoded_prediction}")
    st.write("### Prediction Confidence")
    st.dataframe(proba_df)
