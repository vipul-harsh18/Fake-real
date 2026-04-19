import streamlit as st
import joblib

# Load model & vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# UI
st.title("📰 Fake News Detection App")
st.write("Enter news text below:")

# Input
text = st.text_area("News Text")

# Predict button
if st.button("Predict"):
    if text.strip() != "":
        vec = vectorizer.transform([text])
        result = model.predict(vec)

        if result[0] == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
    else:
        st.warning("Please enter some text")
