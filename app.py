import streamlit as st
from src.Components.pipeline.predict_pipline import PredictPipeline


# 🔹 Page config
st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Prediction System")
st.write("Enter symptoms separated by commas (e.g., fever, cough, headache)")

# 🔹 Input
user_input = st.text_input("Symptoms")

# 🔹 Predict button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter at least one symptom")
    else:
        symptoms = user_input.split(",")

        # 🔹 Prediction
        pipeline = PredictPipeline()
        results = pipeline.predict(symptoms)

        # 🔹 Display results
        st.subheader("Top 3 Predicted Diseases")

        for i, res in enumerate(results, start=1):
            disease = res["disease"]
            prob = res["probability"] * 100

            st.write(f"**{i}. {disease}** — {prob:.2f}%")

        st.success("Prediction completed")