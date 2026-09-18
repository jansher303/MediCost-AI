import streamlit as st
import pandas as pd
import pickle


# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Page title
st.title("MediCost-AI: Medical Insurance Cost Prediction")

st.write(
    "Enter the person's information below to predict their medical insurance cost."
)


# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoking Status",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)


# Prediction button
if st.button("Predict Insurance Cost"):

    # Encode gender
    sex_encoded = 0 if sex == "Female" else 1

    # Encode smoking status
    smoker_encoded = 0 if smoker == "No" else 1

    # Encode region
    region_northwest = 1 if region == "Northwest" else 0
    region_southeast = 1 if region == "Southeast" else 0
    region_southwest = 1 if region == "Southwest" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_encoded],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_encoded],
        "region_northwest": [region_northwest],
        "region_southeast": [region_southeast],
        "region_southwest": [region_southwest]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(
        f"Predicted Medical Insurance Cost: {prediction:.2f}"
    )