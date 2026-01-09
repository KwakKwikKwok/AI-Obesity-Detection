import streamlit as st
import pandas as pd
import joblib

# Load pipeline and label encoder
pipeline = joblib.load("obesity_pipeline.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.title("Obesity Level Prediction App")

st.write("Enter your details below to predict obesity level:")

# Collect user inputs (all 16 features)
gender = st.selectbox("Gender", ["Male", "Female"], help="Enter your gender.")
age = st.number_input("Age", min_value=1, max_value=100, value=25, help="Enter your age.")
height = st.number_input("Height (m)", min_value=1.0, max_value=3.0, value=1.70, help="Enter your weight in meters.")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, help="Enter your weight in kilograms.")

family_history = st.selectbox("Family history with overweight?", ["no", "yes"], help="Select 'yes' if any close family member has a history of overweight or obesity.")
favc = st.selectbox("Do you eat high caloric food frequently (FAVC)?", ["no", "yes"], help="High-calorie foods include fast food, fried foods, and sugary snacks.")
fcvc = st.selectbox("Frequency of vegetable consumption per day (FCVC)", options=[1, 2, 3], index=2, format_func = lambda x: {1: "Never", 2: "Sometimes", 3: "Always"}[x], help="Select how often you usually consume vegetables.")
ncp = st.selectbox("Number of main meals (NCP)", options=[1, 2, 3, 4], format_func = lambda x: {1: "Once a day", 2: "Twice a day", 3: "Three times a day", 4: "More than 3 times a day"}[x], help="Enter the number of main meals you usually eat per day.")
caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"], help="Select how often you eat snacks between main meals.")
smoke = st.selectbox("Do you smoke?", ["no", "yes"], help="Select 'yes' if you currently an active smoker")
CH2O = st.selectbox("Daily water consumption (CH2O)", options=[1, 2, 3], format_func = lambda x: {1: "Less than 1L per day", 2: "1-2 Liters per day", 3: "More than 2 Liters per day"}[x], help="Enter your average daily water intake.")
scc = st.selectbox("Do you monitor your calories (SCC)?", ["no", "yes"], index=1,help="Select 'yes' if you regulary track or control your daily calorie intake.")
faf = st.selectbox("Physical activity frequency (FAF)", options=[0, 1, 2, 3], index=2, format_func = lambda x: {0: "No Physical Activity", 1: "Low", 2: "Moderate", 3: "High (frequent)"}[x], help="Select the level of physical activity that best describes your usual routine.")
tue = st.selectbox("Time using technology devices (TUE)",  options=[0, 1, 2], index=1,format_func = lambda x: {0: "0-2 hours per day", 1: "3-5 hours per day", 2: "More than 5 hours per day"}[x], help="Includes time spent using smartphones, computers, tablets, or watching TV.")
calc = st.selectbox("Consumption of alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"], help="Select how often you consume alcoholic beverages.")
mtrans = st.selectbox("Transportation used (MTRANS)", ["Bike", "Walking", "Motorbike", "Automobile", "Public_Transportation"], help="Select the transportation mode you use most frequently.")

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
