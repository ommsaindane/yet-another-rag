from config.settings import get_settings
from models.chunk import ParentChunk
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import BGEReranker
from storage.parent_store import ParentStore


class RetrievalPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BGEReranker,
        parent_store: ParentStore,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.parent_store = parent_store

    def retrieve(
        self,
        query: str,
        retrieve_k: int | None = None,
        rerank_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ParentChunk]:
        settings = get_settings()
        k_retrieve = retrieve_k if retrieve_k is not None else settings.retrieve_k
        k_rerank = rerank_k if rerank_k is not None else settings.rerank_k

        child_chunks = self.hybrid_retriever.retrieve(
            query=query,
            k=k_retrieve,
        )
        if not child_chunks:
            return []

        reranked_children = self.reranker.rerank(
            query=query,
            chunks=child_chunks,
            top_k=k_rerank,
            score_threshold=score_threshold,
        )

        parents = self.parent_store.get_from_children(reranked_children)

        # Attach the highest child rerank score to its parent chunk metadata
        best_parent_scores: dict[str, float] = {}
        for child in reranked_children:
            score = child.metadata.get("rerank_score", 0.0)
            if child.parent_id not in best_parent_scores:
                best_parent_scores[child.parent_id] = score

        for parent in parents:
            if parent.id in best_parent_scores:
                parent.metadata["retrieval_score"] = best_parent_scores[parent.id]

        return parents