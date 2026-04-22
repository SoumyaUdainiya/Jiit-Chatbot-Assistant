# llm_handler.py
# Uses Groq (llama3-8b) to synthesise a natural answer from retrieved context.
# Store your API key in an environment variable:
#   set GROQ_API_KEY=gsk_...  (Windows)
#   export GROQ_API_KEY=gsk_...  (Mac/Linux)

import os
from groq import Groq

client = Groq(api_key="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")

SYSTEM_PROMPT = """You are JIIT Assistant, a helpful and knowledgeable chatbot for Jaypee Institute of Information Technology, Noida.

Rules:
- Answer ONLY using the provided context. Do NOT invent or assume any facts.
- Be concise and natural. Avoid bullet points unless listing multiple distinct items.
- If the context already answers the question well, just rephrase it clearly.
- If the context contains multiple relevant pieces, synthesise them into one coherent answer.
- Never say "based on the context" or "according to the information provided".
- Speak like a friendly, knowledgeable assistant."""

def generate_response(context: str, user_query: str) -> str:
    """
    Synthesise a natural answer from the retrieved context using Groq LLM.
    
    Args:
        context: All answers retrieved from the matched intent (may be multiple rows)
        user_query: The original user question
    
    Returns:
        A clean, natural language response string
    """
    user_message = f"""Context:
{context}

Student's question: {user_query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.4,   # low = factual, less hallucination
        max_tokens=300
    )

    return response.choices[0].message.content.strip()