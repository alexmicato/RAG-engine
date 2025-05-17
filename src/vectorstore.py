import os
import pickle
import numpy as np
import faiss

# Paths for persistence
INDEX_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'faiss_index')
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
META_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')

# Ensure directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Dimension for text-embedding-3-small
DIM = 384

# Load or initialize FAISS index
def _load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(DIM)

index = _load_index()

# Load or initialize metadata list
if os.path.exists(META_PATH):
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
else:
    metadata = []  # list of dicts: {'id': doc_id, 'text': summary}


def save_state():
    """Persist FAISS index and metadata"""
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump(metadata, f)


def upsert_documents(docs: list[tuple[str, str]], embed_fn) -> None:
    """
    docs: list of (doc_id, summary_text)
    embed_fn: function to embed text into list[float]
    """
    vectors = []
    for doc_id, text in docs:
        vec = embed_fn(text)
        vectors.append(vec)
        metadata.append({'id': doc_id, 'text': text})

    if vectors:
        arr = np.array(vectors, dtype='float32')
        index.add(arr)
        save_state()


def query_top_k(query: str, embed_fn, k: int = 5) -> list[dict]:
    """
    Returns up to k unique metadata entries for the query.
    Each entry is a dict with 'id' and 'text'.
    """
    # If no entries, return empty
    if not metadata:
        return []

    # Embed query
    q_vec = np.array([embed_fn(query)], dtype='float32')
    D, I = index.search(q_vec, k)

    results = []
    seen_ids = set()
    for idx in I[0]:
        # idx == -1 indicates no more entries
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        if entry['id'] in seen_ids:
            continue
        seen_ids.add(entry['id'])
        results.append(entry)
        if len(results) >= min(k, len(metadata)):
            break
    return results

# For embeddings, we’ll use a local SentenceTransformer
from sentence_transformers import SentenceTransformer

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list[float]:
    """
    Creates an embedding vector using sentence-transformers locally.
    """
    return _embed_model.encode(text).tolist()