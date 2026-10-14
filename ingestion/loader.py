from pathlib import Path
from models.document import RawDocument


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
        self.directory = Path(directory).resolve()

        if not self.directory.exists():
            raise FileNotFoundError(f"Directory '{self.directory}' does not exist.")

        if not self.directory.is_dir():
            raise NotADirectoryError(f"'{self.directory}' is not a directory.")

    def load(self) -> list[RawDocument]:
        files = [
            file
            for file in self.directory.rglob("*")
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return self.load_files(files)

    def load_files(self, file_paths: list[Path]) -> list[RawDocument]:
        documents: list[RawDocument] = []

        for file in file_paths:
            resolved = file.resolve()
            if not resolved.exists() or not resolved.is_file():
                continue

            extension = resolved.suffix.lower()
            if extension not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                rel_path = resolved.relative_to(self.directory).as_posix()
            except ValueError:
                rel_path = resolved.name

            documents.append(
                RawDocument(
                    path=resolved,
                    relative_path=rel_path,
                    filename=resolved.name,
                    extension=extension,
                    size=resolved.stat().st_size,
                )
            )

        return sorted(documents, key=lambda doc: doc.relative_path)