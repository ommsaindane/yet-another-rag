from models.chunk import ChildChunk, ParentChunk


class ParentStore:
    def __init__(self):
        self._parents: dict[str, ParentChunk] = {}

    def add(
        self,
        parents: list[ParentChunk],
    ) -> None:

        for parent in parents:
            self._parents[parent.id] = parent

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
