from models.chunk import ChildChunk

from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[ChildChunk]:

        vector_results = self.vector_retriever.retrieve(
            query=query,
            k=k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            k=k,
        )

        rrf_scores: dict[str, float] = {}
        retrieved_chunks: dict[str, ChildChunk] = {}

        #
        # Dense retrieval contribution
        #

        for rank, (chunk, _) in enumerate(
            vector_results,
            start=1,
        ):
            retrieved_chunks[chunk.id] = chunk

            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1 / (
                self.rrf_k + rank
            )

        #
        # Sparse retrieval contribution
        #

        for rank, (chunk, _) in enumerate(
            bm25_results,
            start=1,
        ):
            retrieved_chunks[chunk.id] = chunk

            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1 / (
                self.rrf_k + rank
            )

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        return [retrieved_chunks[chunk_id] for chunk_id in ranked_ids[:k]]
