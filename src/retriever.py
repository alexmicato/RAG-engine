import os
import re
from .vectorstore import query_top_k, embed_text
from .ollama_client import generate_with_ollama

def extract_section(context: str, query: str) -> str:
    """
    Split context into sections at headings (lines ending with ':'),
    and return the text of the section whose heading keyword
    appears in the query. If none match, return full context.
    """
    # Find all headings
    headings = re.findall(r'(?m)^(.*?):\s*$', context)
    for h in headings:
        if h.lower() in query.lower():
            # Capture everything until the next heading or end of text
            pattern = rf'{re.escape(h)}:\s*(.*?)(?=\n^[^:\n]+:|\Z)'
            m = re.search(pattern, context, re.DOTALL | re.MULTILINE)
            if m:
                return m.group(1).strip()
    return context

def answer_query(query: str, top_k: int = 5) -> str:
    # 1) Retrieve top-k documents
    hits = query_top_k(query, embed_fn=embed_text, k=top_k)
    full_context = "\n---\n".join(hit['text'] for hit in hits)

    context = extract_section(full_context, query)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer using ONLY the provided context, concisely. "
        "If the answer isn’t present, reply “No information available.”"
    )

    print(f"DEBUG: Using context section length {len(context)} characters.")

    # 4) Call the model
    try:
        return generate_with_ollama(prompt,
                                    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    except Exception as e:
        print(f"DEBUG: Ollama failed: {e}")
        # (you can keep your regex fallback here)
        return "Sorry, I could not find an exact match in the ingested documents."
