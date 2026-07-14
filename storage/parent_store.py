from dataclasses import asdict
from pathlib import Path
import json

from models.chunk import ParentChunk, ChildChunk


class ParentStore:
    def __init__(
        self,
        store_path: str | Path = "data/indexes/parent_store.json",
    ):
        self.store_path = Path(store_path)
        self.parents: dict[str, ParentChunk] = {}

    def add(
        self,
        parent_chunks: list[ParentChunk],
    ) -> None:

        self.parents = {chunk.id: chunk for chunk in parent_chunks}

    def get_from_children(
        self,
        child_chunks: list[ChildChunk],
    ) -> list[ParentChunk]:

        seen: set[str] = set()
        parents: list[ParentChunk] = []

        for child in child_chunks:
            if child.parent_id in seen:
                continue

            seen.add(child.parent_id)

            parents.append(self.parents[child.parent_id])

        return parents

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
                [asdict(parent) for parent in self.parents.values()],
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

        self.parents = {item["id"]: ParentChunk(**item) for item in data}

    def exists(self) -> bool:
        return self.store_path.exists()
