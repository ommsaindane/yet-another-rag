import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import get_settings
from models.chunk import ChildChunk


class BGEEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ):
        settings = get_settings()
        self.model = SentenceTransformer(
            model_name or settings.embedding_model,
            device=device or settings.embedding_device,
        )
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize_embeddings = normalize_embeddings

    def embed(
        self,
        chunks: list[ChildChunk],
    ) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )