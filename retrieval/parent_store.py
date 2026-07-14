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
        parents: list[ParentChunk],
    ) -> None:

        self._parents = {parent.id: parent for parent in parents}

    def get(
        self,
        parent_id: str,
    ) -> ParentChunk:

        return self._parents[parent_id]

    def get_many(
        self,
        parent_ids: list[str],
    ) -> list[ParentChunk]:

        seen: set[str] = set()
        parents: list[ParentChunk] = []

        for parent_id in parent_ids:
            if parent_id in seen:
                continue

            seen.add(parent_id)

            parents.append(self._parents[parent_id])

        return parents

    def get_from_children(
        self,
        children: list[ChildChunk],
    ) -> list[ParentChunk]:

        return self.get_many([child.parent_id for child in children])

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
