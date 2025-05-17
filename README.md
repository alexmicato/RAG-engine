## Concert Tour RAG Bot
A Python-based Retrieval-Augmented Generation (RAG) service for ingesting, indexing, and querying domain-specific documents about upcoming concert tours (2025–2026).
Overview
This system combines:

📄 Document ingestion with keyword filtering & summarization
🔎 Vector retrieval via FAISS + SentenceTransformers embeddings
🤖 LLM generation (summaries & answers) via local Ollama models
💻 CLI and 🌐 Streamlit UI front-ends


## Design & Approach
1. Ingester

Relevance check by keyword scan
Summarization via Ollama: a concise 3–5 bullet-point summary
Vector indexing: full document text → 384-dim embeddings (all-MiniLM-L6-v2) → FAISS

2. Retriever

Query embedding → FAISS search → top-k hits
Section extraction: split context by headings and select the section matching the user's query keywords
Answer generation via Ollama: "Answer using ONLY the provided context, concisely."

3. Dual Interfaces

CLI (python -m src.cli ingest … / ask …) for power users & scripts
Streamlit UI (streamlit run src/streamlit_app.py) for an interactive web app

4. Model Hosting

Ollama models (GGUF format) live in your user home (~/.ollama/models by default)
Embedding model (all-MiniLM-L6-v2) is downloaded by SentenceTransformers


## Prerequisites

Python 3.8+
Ollama CLI installed & on your PATH
bash# Windows (Chocolatey):
choco install ollama -y

# macOS/Linux:
brew install ollama


Pull Ollama models:
bashollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

## Setup
bashgit clone https://github.com/your-org/concert-tour-rag.git
cd concert-tour-rag
Create & activate virtualenv
bashpython -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
Install dependencies
bashpip install --upgrade pip
pip install -r requirements.txt
Verify Ollama
bashollama --help

## CLI Usage
Ingest a document
bashpython -m src.cli ingest [--doc-id MY_ID] path/to/doc.txt

--doc-id (optional): defaults to filename stem
path/to/doc.txt: UTF-8 or Latin-1 text

Example:
bashpython -m src.cli ingest sample_tour test_data/sample_tour.txt
Ask a question
bashpython -m src.cli ask "Your question here"

## Streamlit Web App
From project root:
bashstreamlit run src/streamlit_app.py

Ingest: Upload a .txt or paste text, click "Ingest Document"
Ask: Type your query, click "Get Answer"


## FAISS Index Management

⚠️ Do NOT commit FAISS index binaries to Git


.gitignore excludes /data/faiss_index/ and *.faiss
Rebuild by re-running your ingest commands or scripting a bulk ingest (ingest-all)


## Testing
bashpytest