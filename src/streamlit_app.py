import streamlit as st
import os
import sys
from pathlib import Path
import hashlib

# Ensure project root is on sys.path so src.* imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingest import ingest_document
from src.retriever import answer_query

st.set_page_config(page_title="Concert Tour RAG", layout="wide")
st.title("🎵 Concert Tour RAG Bot")

st.header("1. Ingest a New Document")

# File uploader for .txt documents
uploaded_file = st.file_uploader("Upload a tour document (.txt)", type=["txt"])

# Or fallback to a text area
doc_text = ""
if not uploaded_file:
    doc_text = st.text_area("Or paste your tour document here:", height=200)
else:
    raw = uploaded_file.read()
    try:
        doc_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        doc_text = raw.decode("latin-1")  # fallback

if st.button("Ingest Document"):
    if not doc_text:
        st.warning("Please upload or paste a document before ingesting.")
    else:
        # Derive a document ID:
        if uploaded_file:
            # use the filename (without extension) as the ID
            doc_id = Path(uploaded_file.name).stem
        else:
            # hash the text as a fallback ID
            doc_id = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:8]

        try:
            summary = ingest_document(doc_text, doc_id=doc_id)
            st.success(f"Document “{doc_id}” ingested successfully!")
            st.write("**Summary:**", summary)
        except ValueError as e:
            st.error(str(e))

st.markdown("---")

st.header("2. Ask a Question")

query = st.text_input("Your question about the tour:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        answer = answer_query(query)
        st.write("**Answer:**", answer)
