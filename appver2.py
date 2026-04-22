# app.py  –  JIIT AI Assistant  (Pure RAG, no intent)
# Flow: query → BERT → top-K similar rows → LLM → answer

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_handler import generate_response
from utils import is_online
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
base_path = os.path.dirname(__file__)

# ── Load BERT ──────────────────────────────────────────────────────────────────
print("Loading BERT model...")
bert = SentenceTransformer("all-MiniLM-L6-v2")

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
data = pd.read_csv(os.path.join(base_path, "college_dataset.csv"))
data["question"] = data["question"].str.lower().str.strip()
data = data.dropna(subset=["question", "answer"]).reset_index(drop=True)
print(f"  {len(data)} rows loaded")

# ── Load/build question embeddings ─────────────────────────────────────────────
emb_path = os.path.join(base_path, "question_embeddings.npy")
if os.path.exists(emb_path):
    print("Loading cached embeddings...")
    question_embeddings = np.load(emb_path)
else:
    print("Encoding dataset (run encode_dataset.py for faster startup)...")
    question_embeddings = bert.encode(data["question"].tolist(), show_progress_bar=True)
    np.save(emb_path, question_embeddings)

print("Ready!\n")

# ── Config ─────────────────────────────────────────────────────────────────────
TOP_K     = 5      # rows to retrieve and send to LLM as context
LOW_CONF  = 0.25   # below this → tell user we don't know
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

REPLACEMENTS = {
    "jit": "jiit", "j iit": "jiit",
    "gi it": "jiit", "machine": "mission", "kyc": "jyc",
}

def normalize(text):
    for wrong, right in REPLACEMENTS.items():
        text = text.replace(wrong, right)
    return text

def retrieve(user_input, top_k=TOP_K):
    """Return deduplicated context from top-k similar rows + best confidence."""
    vec  = bert.encode([user_input])
    sims = cosine_similarity(vec, question_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    best_conf   = float(sims[top_indices[0]])

    # Deduplicate — same answer can appear under multiple question phrasings
    seen, answers = set(), []
    for i in top_indices:
        ans = data.iloc[i]["answer"]
        if ans not in seen:
            seen.add(ans)
            answers.append(ans)

    context = "\n\n".join(answers)
    return context, best_conf

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    body      = request.get_json()
    raw_input = body.get("message", "").strip()

    if not raw_input:
        return jsonify({"response": "Please say something!"})

    user_input = normalize(raw_input.lower())

    if user_input in GREETINGS:
        return jsonify({"response": "Hello! I am JIIT Assistant. How can I help you today?",
                        "confidence": 1.0, "mode": "rule"})

    context, confidence = retrieve(user_input)

    if confidence < LOW_CONF:
        return jsonify({"response": "I don't have information on that. Try rephrasing your question.",
                        "confidence": confidence, "mode": "low-conf"})

    if is_online():
        try:
            response = generate_response(context, raw_input)
            return jsonify({"response": response, "confidence": confidence, "mode": "llm"})
        except Exception as e:
            import traceback; traceback.print_exc()

    # Offline fallback
    vec  = bert.encode([user_input])
    sims = cosine_similarity(vec, question_embeddings)[0]
    best = data.iloc[sims.argmax()]["answer"]
    return jsonify({"response": best, "confidence": confidence, "mode": "offline"})


@app.route("/voice", methods=["POST"])
def voice():
    from speech_to_text import record_audio, transcribe_audio
    try:
        text       = transcribe_audio(record_audio())
        user_input = normalize(text.lower())
        print("Transcribed:", text)

        context, confidence = retrieve(user_input)

        if is_online():
            try:
                response = generate_response(context, text)
                return jsonify({"response": response, "transcribed": text, "mode": "llm"})
            except Exception as e:
                import traceback; traceback.print_exc()

        vec  = bert.encode([user_input])
        sims = cosine_similarity(vec, question_embeddings)[0]
        best = data.iloc[sims.argmax()]["answer"]
        return jsonify({"response": best, "transcribed": text, "mode": "offline"})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
