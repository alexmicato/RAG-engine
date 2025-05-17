# File: src/summarizer.py

import os

from .ollama_client import generate_with_ollama


def summarize_text(text: str) -> str:
    """
    Uses Ollama to produce a short bulleted summary of the tour document.
    """
    prompt = (
        "You are a concise assistant. Read the following concert tour document and "
        "summarize its key information in **3–5 bullet points**, using no more than **75 words** total:\n\n"
        f"{text}\n\n"
        "Summary:\n"
        "- "
    )

    try:
        raw = generate_with_ollama(
            prompt,
            model="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
        )

        bullets = raw.split("Summary:", 1)[-1].strip()
        return bullets
    except Exception:
        # Fallback: first 50 words
        return " ".join(text.split()[:50]) + "…"

