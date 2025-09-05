import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
from keras.saving import register_keras_serializable

# ---------------------------
# 1. Custom RNN for loading older models
# ---------------------------
@register_keras_serializable()
class CompatibleSimpleRNN(SimpleRNN):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)  # drop unsupported arg
        return super().from_config(config)

# ---------------------------
# 2. Load word index
# ---------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# ---------------------------
# 3. Load the model
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
H5_PATH = os.path.join(BASE_DIR, "simple_rnn_imdb.h5")
KERAS_PATH = os.path.join(BASE_DIR, "simple_rnn_imdb.keras")

if os.path.exists(KERAS_PATH):
    model = load_model(KERAS_PATH)
    st.success("✅ Loaded .keras model")
elif os.path.exists(H5_PATH):
    model = load_model(H5_PATH, safe_mode=False, custom_objects={"SimpleRNN": CompatibleSimpleRNN})
    model.save(KERAS_PATH)
    st.success("✅ Converted .h5 to .keras model")
else:
    st.error("❌ Model file not found!")

# ---------------------------
# 4. Helper functions
# ---------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, float(prediction[0][0])

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive / Negative).")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a review text!")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.4f}")
