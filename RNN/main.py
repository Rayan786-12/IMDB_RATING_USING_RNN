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

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import SimpleRNN
import keras

# ✅ Register custom SimpleRNN to handle "time_major"
@keras.saving.register_keras_serializable()
class CompatibleSimpleRNN(SimpleRNN):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)  # drop unsupported arg
        return super().from_config(config)

# Paths
base_dir = os.path.dirname(__file__)
h5_path = os.path.join(base_dir, "simple_rnn_imdb.h5")
keras_path = os.path.join(base_dir, "simple_rnn_imdb.keras")  # will be created

# Load model
if os.path.exists(keras_path):
    model = tf.keras.models.load_model(keras_path)
    print("✅ Loaded converted .keras model")
elif os.path.exists(h5_path):
    model = load_model(
        h5_path,
        safe_mode=False,
        custom_objects={"SimpleRNN": CompatibleSimpleRNN}
    )
    print("✅ Loaded .h5 model with compatibility patch")

    # Save in new format for future
    model.save(keras_path)
    print(f"💾 Converted model saved to {keras_path}")
else:
    raise FileNotFoundError("❌ No model file found!")

print("🚀 Model is ready for predictions!")


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
