from pathlib import Path

from models.chunk import ChildChunk

from embedding.embedder import BGEEmbedder
from embedding.index import Indexer
from vectordb.chroma import ChromaVectorStore


class EmbeddingPipeline:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "documents",
    ):
        self.embedder = BGEEmbedder()

        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.indexer = Indexer(self.vector_store)

    def run(
        self,
        child_chunks: list[ChildChunk],
        overwrite: bool = False,
    ) -> None:

        if not child_chunks:
            return

        if overwrite:
            self.vector_store.delete()

        embeddings = self.embedder.embed(child_chunks)

        self.indexer.index(
            chunks=child_chunks,
            embeddings=embeddings,
        )
