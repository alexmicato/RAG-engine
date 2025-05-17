import pytest
from src.ingest import ingest_document

def test_ingest_returns_summary():
    sample = "The band will play in Paris, London, and New York in autumn 2025."
    summary = ingest_document("test1", sample)
    assert isinstance(summary, str)
    assert len(summary) > 0