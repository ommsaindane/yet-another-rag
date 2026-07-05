from sentence_transformers import CrossEncoder

from models.chunk import ChildChunk


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
        )

    def rerank(
        self,
        query: str,
        chunks: list[ChildChunk],
        top_k: int | None = None,
    ) -> list[ChildChunk]:

        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked = [chunk for chunk, _ in ranked]

        if top_k is not None:
            return reranked[:top_k]

        return reranked
