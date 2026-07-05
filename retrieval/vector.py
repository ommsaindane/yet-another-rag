from models.chunk import ChildChunk

from embedding.embedder import BGEEmbedder
from vectordb.chroma import ChromaVectorStore


class VectorRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: BGEEmbedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[ChildChunk, float]]:

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
        )

        chunks: list[tuple[ChildChunk, float]] = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = ChildChunk(
                id=chunk_id,
                parent_id=metadata["parent_id"],
                document_id=metadata["document_id"],
                text=text,
                metadata=metadata,
            )

            chunks.append((chunk, distance))

        return chunks
