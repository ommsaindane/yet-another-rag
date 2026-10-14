from dataclasses import asdict
from pathlib import Path
import json

from models.chunk import ChildChunk


class ChildStore:
    def __init__(
        self,
        store_path: str | Path = "data/indexes/child_store.json",
    ):
        self.store_path = Path(store_path)
        self._children: dict[str, ChildChunk] = {}

    def add(
        self,
        child_chunks: list[ChildChunk],
    ) -> None:
        """Upserts child chunks into the store without overwriting other documents."""
        for chunk in child_chunks:
            self._children[chunk.id] = chunk

    def get(self) -> list[ChildChunk]:
        return list(self._children.values())

    def delete_by_document_id(
        self,
        document_id: str,
    ) -> int:
        """Removes all child chunks belonging to a given document_id. Returns count removed."""
        to_delete = [
            cid
            for cid, chunk in self._children.items()
            if chunk.document_id == document_id
        ]
        for cid in to_delete:
            del self._children[cid]
        return len(to_delete)

    def clear(self) -> None:
        self._children.clear()

    def count(self) -> int:
        return len(self._children)

    def save(self) -> None:
        self.store_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(
            self.store_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                [asdict(chunk) for chunk in self._children.values()],
                f,
                indent=4,
                ensure_ascii=False,
            )

    def load(self) -> None:
        if not self.store_path.exists():
            raise FileNotFoundError(f"Child store not found: {self.store_path}")
        with open(
            self.store_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
        self._children = {item["id"]: ChildChunk(**item) for item in data}

    def exists(self) -> bool:
        return self.store_path.exists()