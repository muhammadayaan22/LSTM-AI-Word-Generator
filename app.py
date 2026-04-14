import streamlit as st
import pickle
import random

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def generate_text(seed, num_words):
    words = seed.lower().split()

    for _ in range(num_words):
        last_word = words[-1]

        if last_word in model:
            next_word = random.choice(model[last_word])
            words.append(next_word)
        else:
            break

    return " ".join(words)

# UI
st.title("🧠 AI Text Generator (No TensorFlow)")

seed = st.text_input("Enter Seed Text", "once upon a time")
num_words = st.slider("Words to Generate", 1, 50, 10)

if st.button("Generate"):
    result = generate_text(seed, num_words)
    st.write(result)