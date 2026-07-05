from models.chunk import ParentChunk

from retrieval.hybrid import HybridRetriever
from retrieval.parent_store import ParentStore
from retrieval.reranker import BGEReranker


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
        retrieve_k: int = 20,
        rerank_k: int = 5,
    ) -> list[ParentChunk]:

        child_chunks = self.hybrid_retriever.retrieve(
            query=query,
            k=retrieve_k,
        )

        reranked_children = self.reranker.rerank(
            query=query,
            chunks=child_chunks,
            top_k=rerank_k,
        )

        return self.parent_store.get_from_children(reranked_children)
