from models.chunk import ParentChunk, ChildChunk
from models.document import ParsedDocument

from chunking.parent_child import ParentChildChunker


class ChunkingPipeline:
    """
    Orchestrates the document chunking pipeline.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ):
        self.chunker = ParentChildChunker(
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )

    def run(
        self,
        documents: list[ParsedDocument],
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Chunk a collection of parsed documents.
        """

        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        for document in documents:
            parents, children = self.chunker.chunk(document)

            parent_chunks.extend(parents)
            child_chunks.extend(children)

        return parent_chunks, child_chunks

