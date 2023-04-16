import streamlit as st
import requests


def predict(text_input) -> str:
    text = requests.post(
        "http://localhost:3000/text_classifier/predict",
        headers={'content-type': 'text/plain'},
        data=text_input.encode('utf-8')).text
    return int(float(text[2:-2])*100)


text_input = st.text_input("Write your comment")

if st.button(label='Predict'):
    st.write(f"Your comment is toxic with a {predict(text_input)}% chance")
