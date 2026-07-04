from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RawDocument:
    path: Path
    filename: str
    extension: str
    size: int


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".md",
        ".txt",
        ".html",
    }

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

        if not self.directory.exists():
            raise FileNotFoundError(f"Directory '{self.directory}' does not exist.")

        if not self.directory.is_dir():
            raise NotADirectoryError(f"'{self.directory}' is not a directory.")

    def load(self) -> list[RawDocument]:
        documents: list[RawDocument] = []

        for file in self.directory.rglob("*"):
            if not file.is_file():
                continue

            extension = file.suffix.lower()

            if extension not in self.SUPPORTED_EXTENSIONS:
                continue

            documents.append(
                RawDocument(
                    path=file.resolve(),
                    filename=file.name,
                    extension=extension,
                    size=file.stat().st_size,
                )
            )

        return sorted(documents, key=lambda doc: doc.path)

