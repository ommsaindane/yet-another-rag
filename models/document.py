from dataclasses import dataclass
from pathlib import Path
from docling_core.types.doc import DoclingDocument


@dataclass(slots=True)
class RawDocument:
    path: Path
    filename: str
    extension: str
    size: int


@dataclass(slots=True)
class ParsedDocument:
    id: str
    markdown: str
    metadata: dict
    docling_document: DoclingDocument
