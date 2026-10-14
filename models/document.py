from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RawDocument:
    path: Path
    relative_path: str
    filename: str
    extension: str
    size: int


@dataclass(slots=True)
class ParsedDocument:
    id: str
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)