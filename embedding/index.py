from numpy.typing import NDArray

from models.chunk import ChildChunk
from vectordb.chroma import ChromaVectorStore


class Indexer:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def index(
        self,
        chunks: list[ChildChunk],
        embeddings: NDArray,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks should match the number of embeddings.")

        self.vector_store.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "parent_id": chunk.parent_id,
                    "document_id": chunk.document_id,
                }
                for chunk in chunks
            ],
        )
