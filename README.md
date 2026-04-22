# JIIT AI Assistant (RAG-based Chatbot)

## Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot designed for answering queries related to JIIT. It uses BERT embeddings for semantic search and an LLM API for generating responses.

## Features

* Semantic search using Sentence-BERT
* Top-K retrieval for context building
* LLM-based response generation
* Offline fallback when API is unavailable
* Voice input support (speech-to-text)
* Flask-based web interface

## Project Structure

```
.
├── app.py
├── llm_handler.py
├── utils.py
├── speech_to_text.py
├── college_dataset.csv
├── question_embeddings.npy
├── templates/
│   └── index.html
└── __pycache__/   (ignored)
```

## Installation

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

## Running the App

```
python app.py
```

The app will run on:

```
http://127.0.0.1:5000/
```

## How It Works

1. User query is normalized
2. Query is encoded using BERT
3. Cosine similarity is used to retrieve top-K relevant entries
4. Retrieved context is sent to LLM API
5. If offline, best matching answer is returned

## Configuration

* `TOP_K`: Number of retrieved results
* `LOW_CONF`: Confidence threshold for fallback

## Notes

* Ensure `question_embeddings.npy` exists for faster startup
* Do not upload `__pycache__` folder
* Keep API keys secure (use environment variables)

## Future Improvements

* Better ranking strategy (re-ranking models)
* UI enhancements
* Multi-language support
* Fine-tuned domain model
