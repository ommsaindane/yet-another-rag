from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app import app
from api.dependencies import graph_dependency
from models.chunk import ParentChunk


def test_health_and_query_endpoints():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "query": "What is CRAG?",
        "search_query": "What is CRAG?",
        "retrieved_parents": [],
        "relevant_parents": [
            ParentChunk(
                id="p1",
                document_id="d1",
                text="CRAG stands for Corrective RAG.",
                metadata={
                    "filename": "paper.pdf",
                    "section_title": "Introduction",
                    "retrieval_score": 0.94,
                },
            )
        ],
        "context_is_relevant": True,
        "web_results": [],
        "context": "\nCRAG stands for Corrective RAG.",
        "retrieved_contexts": ["CRAG stands for Corrective RAG."],
        "answer": "CRAG stands for Corrective RAG.",
    }

    app.dependency_overrides[graph_dependency] = lambda: mock_graph
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # 2. Standard /query endpoint
    res_query = client.post("/query", json={"query": "What is CRAG?"})
    assert res_query.status_code == 200
    data = res_query.json()
    assert data["web_fallback"] is False
    assert data["sources"][0]["source_type"] == "local"
    assert data["sources"][0]["document"] == "paper.pdf"
    assert data["sources"][0]["section"] == "Introduction"

    app.dependency_overrides.clear()