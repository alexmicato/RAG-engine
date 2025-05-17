# File: src/summarizer.py

import os
import re
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
            model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
        )

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        bullets = []
        for ln in lines:
            m = re.match(r"^(\d+[.)]|[-•●])\s*(.+)", ln)  # ①., 1), -, •, ●
            if m:
                bullets.append(m.group(2).strip())
        if not bullets:
            bullets = [" ".join(text.split()[:50]) + "…"]
        return "- " + "\n- ".join(bullets[:5])

    except Exception:
        # Fallback: first 50 words
        return " ".join(text.split()[:50]) + "…"

