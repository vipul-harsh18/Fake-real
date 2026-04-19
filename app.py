import streamlit as st
import pickle

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

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
