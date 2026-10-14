import re
from rank_bm25 import BM25Okapi
from models.chunk import ChildChunk


_TOKEN_PATTERN = re.compile(r"\b[a-z0-9_]{2,}\b")

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "him", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
        "no", "not", "of", "on", "or", "our", "out", "she", "so", "some",
        "such", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "too", "up", "us", "very", "was",
        "we", "were", "what", "when", "where", "which", "who", "whom",
        "why", "will", "with", "would", "you", "your",
    }
)


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[ChildChunk] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Lowercases text, extracts alphanumeric tokens, and removes common stopwords.
        Falls back to unfiltered tokens if the query consists solely of stopwords.
        """
        tokens = _TOKEN_PATTERN.findall(text.lower())
        filtered = [t for t in tokens if t not in _STOPWORDS]
        return filtered if filtered else tokens

    def index(
        self,
        chunks: list[ChildChunk],
    ) -> None:
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        corpus = [self.tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[ChildChunk, float]]:
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Filter out zero-score chunks so non-matching documents don't pollute RRF
        scored_chunks = [
            (chunk, float(score))
            for chunk, score in zip(self.chunks, scores)
            if score > 0.0
        ]

        ranked = sorted(
            scored_chunks,
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:k]