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
        self.children: list[ChildChunk] = []

    def add(
        self,
        child_chunks: list[ChildChunk],
    ) -> None:

        self.children = child_chunks

    def get(self) -> list[ChildChunk]:

        return self.children

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
                [asdict(chunk) for chunk in self.children],
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

        self.children = [ChildChunk(**item) for item in data]

    def exists(self) -> bool:

        return self.store_path.exists()
