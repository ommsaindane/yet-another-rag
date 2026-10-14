from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParentChunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChildChunk:
    id: str
    parent_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)