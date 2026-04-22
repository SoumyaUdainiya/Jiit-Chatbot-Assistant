# encode_dataset.py
# Run this ONCE after any change to college_dataset.csv
# Outputs:
#   question_embeddings.npy  – per-row embeddings  (fallback)
#   intent_centroids.npy     – average embedding per intent (primary matching)
#   intent_labels.npy        – ordered intent names matching centroid rows

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
data = pd.read_csv("college_dataset.csv")
data["question"] = data["question"].str.lower().str.strip()
data = data.dropna(subset=["question", "intent", "answer"]).reset_index(drop=True)

print(f"Loaded {len(data)} rows across {data['intent'].nunique()} intents.")

print("Loading BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Encode every row
print("Encoding all questions...")
all_embeddings = model.encode(data["question"].tolist(), show_progress_bar=True)
np.save("question_embeddings.npy", all_embeddings)
print(f"  Saved question_embeddings.npy  shape={all_embeddings.shape}")

# 2. Build centroid per intent
print("Building intent centroids...")
intents = data["intent"].unique()
centroids = []

for intent in intents:
    mask = data["intent"] == intent
    centroid = all_embeddings[mask].mean(axis=0)
    centroids.append(centroid)

centroids_arr = np.array(centroids)
np.save("intent_centroids.npy", centroids_arr)
np.save("intent_labels.npy", np.array(intents))

print(f"  Saved intent_centroids.npy  shape={centroids_arr.shape}")
print(f"  Intents: {list(intents)}")
print("\nDone! Run app.py next.")