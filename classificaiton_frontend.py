import streamlit as st
import requests

st.title("Drug Review Classifier")

API_URL = "http://localhost:6000/classification"

text = st.text_area("Enter the drug review:", height=200)

if st.button('Classify Review'):
    with st.spinner("Classifying..."):
        response = requests.post(API_URL, json={'text': text})
        if response.status_code == 200:
            classification_results = response.json()
            print(classification_results)
            st.subheader("Condition Classified: ")
            st.subheader(classification_results)
            st.write("---")
        else:
            st.error("Error during classification.")
