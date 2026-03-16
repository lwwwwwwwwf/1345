import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("academic_warning_model.pkl")

st.title("Academic Warning Prediction")

st.write("Nhập thông tin sinh viên")

gpa = st.number_input("GPA", 0.0, 4.0, 2.5)
credits = st.number_input("Credits Registered", 0, 30, 15)
absences = st.number_input("Absences", 0, 50, 5)

major = st.selectbox(
    "Major",
    ["IT","Business","Economics","Engineering"]
)

data = pd.DataFrame({
    "gpa":[gpa],
    "credits":[credits],
    "absences":[absences],
    "major":[major]
})

if st.button("Predict"):

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Student may receive Academic Warning")
    else:
        st.success("✅ Student is safe")
