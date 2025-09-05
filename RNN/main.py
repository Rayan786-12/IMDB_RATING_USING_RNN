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
import os
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

# 🔧 Custom wrapper to ignore unsupported "time_major" argument
class CompatibleSimpleRNN(SimpleRNN):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)  # drop unsupported arg
        return super().from_config(config)

# Path to the old H5 model
model_path = os.path.join(os.path.dirname(__file__), "simple_rnn_imdb.h5")

# Path to the converted SavedModel
converted_path = os.path.join(os.path.dirname(__file__), "converted_model")

# If already converted, load the new format
if os.path.exists(converted_path):
    model = tf.keras.models.load_model(converted_path)
    print("✅ Loaded converted SavedModel.")
else:
    # Load old H5 model with compatibility fix
    model = load_model(
        model_path,
        safe_mode=False,
        custom_objects={"SimpleRNN": CompatibleSimpleRNN}
    )
    print("✅ Loaded old H5 model with compatibility patch.")

    # Save in new SavedModel format for future use
    model.save(converted_path, save_format="tf")
    print(f"💾 Converted model saved to {converted_path}")

# At this point, `model` is ready to use


# Pass safe_mode and custom_objects here


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
