# File: src/ingest.py

from .summarizer import summarize_text
from .vectorstore import upsert_documents, embed_text
import hashlib
from typing import Optional

# Simple keyword relevancy check
def is_relevant(text: str) -> bool:
    keywords = [
        "concert", "tour", "venue", "show", "performance",
        "schedule", "performer", "logistical", "stage", "arena", "stadium",
        "band", "play", "guest", "tickets"
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)

def ingest_document(text: str, doc_id: Optional[str] = None) -> str:
    """
    1) Verify relevance.
    2) Generate a summary for user confirmation.
    3) Index the FULL text (so every line is searchable).
    """
    if not is_relevant(text):
        raise ValueError(
            "Sorry, I cannot ingest documents with other themes. "
            "Please provide a concert tour-related document."
        )

    # 0) If no doc_id provided, generate one (e.g. a hash of the text)
    if doc_id is None:
        doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    # 1. Summary for the user
    summary = summarize_text(text)

    # 2. Upsert full text into FAISS
    upsert_documents([(doc_id, text)], embed_fn=embed_text)

    return summary
