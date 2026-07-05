from sentence_transformers import SentenceTransformer
import numpy as np

from models.chunk import ChildChunk


class BGEEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def embed(
        self,
        chunks: list[ChildChunk],
    ) -> np.ndarray:

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings
