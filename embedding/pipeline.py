from pathlib import Path
from config.settings import get_settings
from embedding.embedder import BGEEmbedder
from embedding.index import Indexer
from models.chunk import ChildChunk
from vectordb.chroma import ChromaVectorStore


class EmbeddingPipeline:
    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str | None = None,
        embedder: BGEEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        settings = get_settings()
        persist_dir = persist_directory or settings.chroma_dir
        col_name = collection_name or settings.collection_name

        self.embedder = embedder or BGEEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=col_name,
        )
        self.indexer = Indexer(self.vector_store)

    def run(
        self,
        child_chunks: list[ChildChunk],
        overwrite: bool = False,
    ) -> None:
        if overwrite:
            self.vector_store.delete()

        if not child_chunks:
            return

        embeddings = self.embedder.embed(child_chunks)
        self.indexer.index(
            chunks=child_chunks,
            embeddings=embeddings,
        )