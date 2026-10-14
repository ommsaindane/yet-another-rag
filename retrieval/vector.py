from embedding.embedder import BGEEmbedder
from models.chunk import ChildChunk
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
        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
        )

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        chunks: list[tuple[ChildChunk, float]] = []

        for chunk_id, text, metadata, distance in zip(
            ids_list,
            docs_list,
            metas_list,
            dists_list,
        ):
            meta_copy = dict(metadata) if metadata else {}
            parent_id = meta_copy.pop("parent_id", "")
            document_id = meta_copy.pop("document_id", "")

            chunk = ChildChunk(
                id=chunk_id,
                parent_id=parent_id,
                document_id=document_id,
                text=text,
                metadata=meta_copy,
            )
            chunks.append((chunk, float(distance)))

        return chunks