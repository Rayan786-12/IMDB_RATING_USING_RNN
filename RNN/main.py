import numpy as np
import  tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

## Load the imdb dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}
## Load the pret rained model with Relu Activation function
# model=load_model('C:\Machine_learning\RNN\RNN\simple_rnn_imdb.h5')
import os
from tensorflow.keras.models import load_model

model_path = os.path.join(os.getcwd(), 'simple_rnn_imdb.h5')
model = load_model(model_path)
# model.summary()
## Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review]    )   
## Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=pad_sequences([encoded_review],maxlen=500)
    return padded_review
## PRediction function
def predict_sentiment(review):
    processed_review=preprocess_text(review)
    prediction=model.predict(processed_review)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]
##streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to predict its sentiment (positive or negative).')
#User input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)  

    ##Make prediction

    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    #Display the result
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Confidence Score:{prediction[0][0]:.4f}")
else:
    st.write('Please enter a movie review and click "Classify" to see the sentiment prediction.')
