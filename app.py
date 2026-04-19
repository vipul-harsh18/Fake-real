import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detector")

# User input
input_text = st.text_area("Enter News Text")

# Button
if st.button("Check"):
    if input_text.strip() != "":
        vec = vectorizer.transform([input_text])
        result = model.predict(vec)

        if result[0] == 0:
            st.error("Fake News ❌")
        else:
            st.success("Real News ✅")
    else:
        st.warning("Please enter some text")