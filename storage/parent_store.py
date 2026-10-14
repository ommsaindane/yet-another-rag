from dataclasses import asdict
from pathlib import Path
import json

from models.chunk import ChildChunk, ParentChunk


class ParentStore:
    def __init__(
        self,
        store_path: str | Path = "data/indexes/parent_store.json",
    ):
        self.store_path = Path(store_path)
        self._parents: dict[str, ParentChunk] = {}

    def add(
        self,
        parent_chunks: list[ParentChunk],
    ) -> None:
        """Upserts parent chunks into the in-memory store without wiping existing ones."""
        for chunk in parent_chunks:
            self._parents[chunk.id] = chunk

    def get(
        self,
        parent_id: str,
    ) -> ParentChunk | None:
        return self._parents.get(parent_id)

    def get_many(
        self,
        parent_ids: list[str],
    ) -> list[ParentChunk]:
        seen: set[str] = set()
        parents: list[ParentChunk] = []
        for parent_id in parent_ids:
            if parent_id in seen:
                continue
            parent = self._parents.get(parent_id)
            if parent is not None:
                seen.add(parent_id)
                parents.append(parent)
        return parents

    def get_from_children(
        self,
        child_chunks: list[ChildChunk],
    ) -> list[ParentChunk]:
        return self.get_many([child.parent_id for child in child_chunks])

    def delete_by_document_id(
        self,
        document_id: str,
    ) -> int:
        """Removes all parent chunks associated with a document_id. Returns count removed."""
        to_delete = [
            pid
            for pid, chunk in self._parents.items()
            if chunk.document_id == document_id
        ]
        for pid in to_delete:
            del self._parents[pid]
        return len(to_delete)

    def clear(self) -> None:
        self._parents.clear()

    def count(self) -> int:
        return len(self._parents)

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
                [asdict(parent) for parent in self._parents.values()],
                f,
                indent=4,
                ensure_ascii=False,
            )

    def load(self) -> None:
        if not self.store_path.exists():
            raise FileNotFoundError(f"Parent store not found: {self.store_path}")
        with open(
            self.store_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
        self._parents = {item["id"]: ParentChunk(**item) for item in data}

    def exists(self) -> bool:
        return self.store_path.exists()