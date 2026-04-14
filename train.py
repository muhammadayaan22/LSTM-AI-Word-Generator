import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load dataset
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# Padding
max_seq_len = max([len(seq) for seq in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
)

# Split features & labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_seq_len-1),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=50, verbose=1)

# Save model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_seq_len-1),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=50, verbose=1)

# Save model
model.save("model.h5")


# Load model
def generate_text(seed_text, next_words, model, tokenizer, max_seq_len):
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
