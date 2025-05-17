import os
import re
from .vectorstore import query_top_k, embed_text
from .ollama_client import generate_with_ollama


def extract_section(context: str, query: str) -> str:
    """
    Return only the document sections that mention the *specific artist / band*
    named in the query.  We treat capitalised words in the query as 'primary'
    search terms and require at least one of them to appear in a section.
    """

    import re

    # Proper-nouns (capitalised words) in the user question → likely artist names
    primary_terms = [w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", query)]

    # Generic terms (tour, autumn…) – we keep them but give them less weight
    generic_terms = [w for w in query.lower().split() if len(w) > 3]

    sections = re.split(r"(?m)^(.*?):\s*$", context)          # heading + content
    section_pairs = [
        (sections[i], sections[i + 1]) for i in range(1, len(sections), 2)
        if i + 1 < len(sections)
    ]

    kept = []
    for heading, content in section_pairs:
        joined = f"{heading} {content}".lower()

        # ➊ If we detected artist-like words, require at least one of them
        if primary_terms and not any(pt in joined for pt in primary_terms):
            continue

        # ➋ Otherwise (or additionally) insist on at least one generic hit
        if not any(gt in joined for gt in generic_terms):
            continue

        kept.append(f"{heading}:\n{content}")

    return "\n\n".join(kept) if kept else context


def answer_query(query: str, top_k: int = 5) -> str:
    # 1) Retrieve top-k documents
    hits = query_top_k(query, embed_fn=embed_text, k=top_k)
    full_context = "\n---\n".join(hit['text'] for hit in hits)

    context = extract_section(full_context, query)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer using ONLY the provided context. If the question asks about a list of items, "
        "include every item that appears **in the context** and nothing else. "
        "If none of the context answers the question, reply exactly: No information available."
    )

    print(f"DEBUG: Using context section length {len(context)} characters.")

    import re
    bullets = re.findall(r"^\s*-\s*(.+?\(\w{3}\s+\d{1,2}\))", context, flags=re.M)
    if bullets:
        seen = set()
        uniq = [b for b in bullets if not (b in seen or seen.add(b))]
        return ", ".join(uniq) + "."

    try:
        return generate_with_ollama(prompt,
                                    model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    except Exception as e:
        print(f"DEBUG: Ollama failed: {e}")
        # (you can keep your regex fallback here)
        return "Sorry, I could not find an exact match in the ingested documents."
