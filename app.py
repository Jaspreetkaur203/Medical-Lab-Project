import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("medical_lab_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Medical Lab Prediction",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Medical Laboratory Test Prediction")
st.write("Predict CBC or Biochemistry test panel")

st.divider()

# User inputs
age = st.number_input(
    "Enter Patient Age",
    min_value=0,
    max_value=120,
    value=25
)

sex = st.selectbox(
    "Select Gender",
    ["Male", "Female"]
)

department = st.selectbox(
    "Select Department",
    ["General Medicine", "Other"]
)

if st.button("Predict Test Panel"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "department": department
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["sex", "department"],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Test Panel: {prediction[0]}")

    st.info("This is an educational ML demonstration, not a clinical recommendation.")
