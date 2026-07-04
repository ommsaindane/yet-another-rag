from uuid import uuid4

from chunking.splitter import TextSplitter
from models.chunk import ParentChunk, ChildChunk
from models.document import ParsedDocument


class ParentChildChunker:
    """
    Generates hierarchical parent-child chunks from a parsed document.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = TextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )

        self.child_splitter = TextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

    def chunk(
        self,
        document: ParsedDocument,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Creates parent and child chunks from a parsed document.
        """

        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        parent_texts = self.parent_splitter.split(document.markdown)

        for parent_text in parent_texts:
            parent_id = str(uuid4())

            parent_chunk = ParentChunk(
                id=parent_id,
                document_id=document.id,
                text=parent_text,
                metadata=document.metadata.copy(),
            )

            parent_chunks.append(parent_chunk)

            child_texts = self.child_splitter.split(parent_text)

            for child_text in child_texts:
                child_chunk = ChildChunk(
                    id=str(uuid4()),
                    parent_id=parent_id,
                    document_id=document.id,
                    text=child_text,
                    metadata=document.metadata.copy(),
                )

                child_chunks.append(child_chunk)

        return parent_chunks, child_chunks
