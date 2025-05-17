# File: src/ingest.py

from .summarizer import summarize_text
from .vectorstore import upsert_documents, embed_text

# Simple keyword relevancy check
def is_relevant(text: str) -> bool:
    keywords = [
        "concert", "tour", "venue", "show", "performance",
        "schedule", "performer", "logistical", "stage", "arena", "stadium",
        "band", "play", "guest", "tickets"
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)

def ingest_document(doc_id: str, text: str) -> str:
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

    # 1. Summary for the user
    summary = summarize_text(text)

    # 2. Upsert full text into FAISS
    upsert_documents([(doc_id, text)], embed_fn=embed_text)

    return summary
