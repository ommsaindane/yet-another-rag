from dataclasses import dataclass


@dataclass(slots=True)
class ParentChunk:
    id: str
    document_id: str
    text: str
    metadata: dict


@dataclass(slots=True)
class ChildChunk:
    id: str
    parent_id: str
    document_id: str
    text: str
    metadata: dict
