import pytest
from src.ingest import ingest_document
from src.retriever import answer_query


def test_retriever_can_find_city():
    text = "Our 2026 tour includes a show in Tokyo."
    ingest_document("test2", text)
    ans = answer_query("Where is the tour?")
    assert "Tokyo" in ans