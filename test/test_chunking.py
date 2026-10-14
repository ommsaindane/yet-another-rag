from chunking.parent_child import ParentChildChunker
from models.document import ParsedDocument


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