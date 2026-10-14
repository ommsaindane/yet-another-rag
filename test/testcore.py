from pathlib import Path
from unittest.mock import MagicMock

from agent.graph import CRAGGraph
from chunking.parent_child import ParentChildChunker
from models.chunk import ChildChunk, ParentChunk
from models.document import ParsedDocument
from models.web import WebResult
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from storage.child_store import ChildStore
from storage.manifest import Manifest
from storage.parent_store import ParentStore


def test_manifest_incremental_diff(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest_file = tmp_path / "manifest.json"

    doc_a = raw_dir / "a.md"
    doc_b = raw_dir / "b.txt"
    doc_a.write_text("Hello world", encoding="utf-8")
    doc_b.write_text("Second file", encoding="utf-8")

    manifest = Manifest(raw_directory=raw_dir, manifest_path=manifest_file)

    # Initial diff should detect 2 added files
    diff1 = manifest.get_diff()
    assert len(diff1.added) == 2
    assert len(diff1.modified) == 0
    assert len(diff1.removed) == 0
    manifest.save()

    # No changes after saving
    assert not manifest.get_diff().has_changes

    # Modify a.md and delete b.txt
    doc_a.write_text("Hello updated world", encoding="utf-8")
    doc_b.unlink()

    diff2 = manifest.get_diff()
    assert len(diff2.added) == 0
    assert [p.name for p in diff2.modified] == ["a.md"]
    assert diff2.removed == ["b.txt"]


def test_structure_aware_parent_child_chunking():
    markdown = (
        "# System Architecture\n"
        "The system uses hybrid retrieval.\n\n"
        "## Dense Vector Search\n"
        "Embeddings are computed using BGE-small."
    )
    doc = ParsedDocument(
        id="doc123",
        markdown=markdown,
        metadata={"filename": "arch.md"},
    )
    chunker = ParentChildChunker(
        parent_chunk_size=100,
        parent_chunk_overlap=10,
        child_chunk_size=60,
        child_chunk_overlap=5,
    )
    parents, children = chunker.chunk(doc)

    assert len(parents) >= 1
    assert len(children) >= 1
    assert parents[0].id == "doc123_p0"
    assert children[0].id == "doc123_p0_c0"
    assert "section_title" in parents[0].metadata


def test_parent_and_child_store_granular_deletion(tmp_path: Path):
    p_store = ParentStore(store_path=tmp_path / "parents.json")
    c_store = ChildStore(store_path=tmp_path / "children.json")

    p_store.add(
        [
            ParentChunk(id="d1_p0", document_id="d1", text="Doc 1 parent"),
            ParentChunk(id="d2_p0", document_id="d2", text="Doc 2 parent"),
        ]
    )
    c_store.add(
        [
            ChildChunk(id="d1_c0", parent_id="d1_p0", document_id="d1", text="Doc 1 child"),
            ChildChunk(id="d2_c0", parent_id="d2_p0", document_id="d2", text="Doc 2 child"),
        ]
    )

    assert p_store.delete_by_document_id("d1") == 1
    assert c_store.delete_by_document_id("d1") == 1
    assert p_store.get("d1_p0") is None
    assert p_store.get("d2_p0") is not None
    assert len(c_store.get()) == 1


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

    # Punctuation and casing should still match "ChromaDB"
    results = bm25.retrieve("How does chromadb work?", k=5)
    assert len(results) == 1
    assert results[0][0].id == "c1"

    # Unrelated query should return empty list rather than 0.0 score noise
    unrelated = bm25.retrieve("quantum gravity tensor", k=5)
    assert unrelated == []


def test_hybrid_rrf_ranking():
    c1 = ChildChunk(id="c1", parent_id="p1", document_id="d1", text="Chunk 1")
    c2 = ChildChunk(id="c2", parent_id="p2", document_id="d1", text="Chunk 2")

    mock_vector = MagicMock()
    mock_bm25 = MagicMock()

    # c2 is rank 1 in both, c1 is rank 2 in vector only
    mock_vector.retrieve.return_value = [(c2, 0.1), (c1, 0.4)]
    mock_bm25.retrieve.return_value = [(c2, 5.2)]

    hybrid = HybridRetriever(
        vector_retriever=mock_vector,
        bm25_retriever=mock_bm25,
        rrf_k=60,
    )
    ranked = hybrid.retrieve("test query", k=2)
    assert [c.id for c in ranked] == ["c2", "c1"]


def test_crag_graph_web_fallback_routing():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = [
        ParentChunk(
            id="p1",
            document_id="d1",
            text="Irrelevant local chunk",
            metadata={"filename": "local.pdf"},
        )
    ]

    mock_llm = MagicMock()
    # Grade returns False -> rewrite_query returns "optimized query" -> generate returns final answer
    mock_llm.generate_structured.side_effect = Exception("Use text fallback")
    mock_llm.generate.side_effect = [
        "NO",
        "optimized web search query",
        "Answer from web context.",
    ]

    mock_prompts = MagicMock()
    mock_prompts.get.return_value = ("sys", "user")

    mock_web = MagicMock()
    mock_web.search.return_value = [
        WebResult(title="Web Doc", url="https://example.com", snippet="Web fact")
    ]

    graph = CRAGGraph(
        retrieval=mock_retrieval,
        llm=mock_llm,
        prompts=mock_prompts,
        web_search=mock_web,
    )
    state = graph.invoke("What is the latest version?")

    assert state["context_is_relevant"] is False
    assert state["relevant_parents"] == []
    assert state["search_query"] == "optimized web search query"
    assert "https://example.com" in state["context"]
    assert "Irrelevant local chunk" not in state["context"]
    assert state["answer"] == "Answer from web context."