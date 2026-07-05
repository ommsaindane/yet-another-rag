from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from numpy.typing import NDArray


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    def add(
        self,
        ids: list[str],
        embeddings: NDArray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: NDArray,
        k: int = 5,
    ) -> dict[str, Any]:

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )

    def count(self) -> int:
        return self.collection.count()

    def delete(self) -> None:

        self.client.delete_collection(self.collection.name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={
                "hnsw:space": "cosine",
            },
        )
