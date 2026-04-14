import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer
model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = 20  # must match training

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# UI
st.title("🧠 AI Text Generator (LSTM)")

seed = st.text_input("Enter Seed Text", "Once upon a time")
num_words = st.slider("Words to Generate", 1, 50, 10)

if st.button("Generate"):
    result = generate_text(seed, num_words)
    st.write(result)