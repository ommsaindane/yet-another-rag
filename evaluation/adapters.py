from typing import List

from ragas.embeddings import BaseRagasEmbeddings

from embedding.embedder import BGEEmbedder
from models.chunk import ChildChunk


class BGEEmbeddingAdapter(BaseRagasEmbeddings):
    def __init__(self):
        self.embedder = BGEEmbedder()

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text).tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        chunks = [
            ChildChunk(
                id=str(i),
                parent_id="",
                document_id="",
                text=text,
                metadata={},
            )
            for i, text in enumerate(texts)
        ]

        embeddings = self.embedder.embed(chunks)

        return embeddings.tolist()
