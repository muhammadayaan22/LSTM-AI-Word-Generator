import pickle
import random

# Load dataset
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

words = text.split()

# Build word dictionary
model = {}

for i in range(len(words)-1):
    word = words[i]
    next_word = words[i+1]

    if word not in model:
        model[word] = []

    model[word].append(next_word)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")