from rank_bm25 import BM25Okapi

from models.chunk import ChildChunk


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunkd: list[ChildChunk] = []

    def index(
        self,
        chunks: list[ChildChunk],
    ) -> None:
        self.chunks = chunks

        corpus = [chunk.text.split() for chunk in chunks]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[ChildChunk, float]]:

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = query.split()

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:k]
