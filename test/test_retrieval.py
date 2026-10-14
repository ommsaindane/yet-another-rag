from unittest.mock import MagicMock
from models.chunk import ChildChunk
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever


def test_bm25_tokenization_and_zero_score_filtering():
    bm25 = BM25Retriever()
    chunks = [
        ChildChunk(
            id="c1",
            parent_id="p1",
            document_id="d1",
            text="Corrective RAG uses LangGraph and ChromaDB.",
        ),
        ChildChunk(
            id="c2",
            parent_id="p2",
            document_id="d2",
            text="Photosynthesis occurs in chloroplasts of plants.",
        ),
    ]
    bm25.index(chunks)

    results = bm25.retrieve("How does chromadb work?", k=5)
    assert len(results) == 1
    assert results[0][0].id == "c1"

    unrelated = bm25.retrieve("quantum gravity tensor", k=5)
    assert unrelated == []


def test_hybrid_rrf_ranking():
    c1 = ChildChunk(id="c1", parent_id="p1", document_id="d1", text="Chunk 1")
    c2 = ChildChunk(id="c2", parent_id="p2", document_id="d1", text="Chunk 2")

    mock_vector = MagicMock()
    mock_bm25 = MagicMock()

    mock_vector.retrieve.return_value = [(c2, 0.1), (c1, 0.4)]
    mock_bm25.retrieve.return_value = [(c2, 5.2)]

    hybrid = HybridRetriever(
        vector_retriever=mock_vector,
        bm25_retriever=bm25 if (bm25 := mock_bm25) else mock_bm25,
        rrf_k=60,
    )
    ranked = hybrid.retrieve("test query", k=2)
    assert [c.id for c in ranked] == ["c2", "c1"]