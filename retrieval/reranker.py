import numpy as np
from sentence_transformers import CrossEncoder
from models.chunk import ChildChunk


class BGEReranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        from config.settings import get_settings
        settings = get_settings()
        self.model = CrossEncoder(
            model_name or settings.reranker_model,
            trust_remote_code=True,
            device=device or settings.reranker_device,
        )

    def rerank(
        self,
        query: str,
        chunks: list[ChildChunk],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ChildChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=np.float64)

        # Normalize raw logits to [0, 1] via sigmoid
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores))

        scored_chunks: list[tuple[ChildChunk, float]] = []
        for chunk, prob in zip(chunks, probabilities):
            score = float(prob)
            if score_threshold is not None and score < score_threshold:
                continue
            chunk.metadata["rerank_score"] = round(score, 4)
            scored_chunks.append((chunk, score))

        ranked = sorted(
            scored_chunks,
            key=lambda x: x[1],
            reverse=True,
        )

        reranked = [chunk for chunk, _ in ranked]
        if top_k is not None:
            return reranked[:top_k]
        return reranked