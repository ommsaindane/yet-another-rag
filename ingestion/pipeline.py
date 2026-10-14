from pathlib import Path
from ingestion.loader import DocumentLoader
from ingestion.parser import DoclingParser
from models.document import ParsedDocument


class IngestionPipeline:
    def __init__(self, data_dir: str | Path, do_ocr: bool = False):
        self.loader = DocumentLoader(data_dir)
        self.parser = DoclingParser(do_ocr=do_ocr)

    def run(self, file_paths: list[Path] | None = None) -> list[ParsedDocument]:
        raw_documents = (
            self.loader.load_files(file_paths)
            if file_paths is not None
            else self.loader.load()
        )

        parsed_documents: list[ParsedDocument] = []
        for raw_document in raw_documents:
            try:
                parsed = self.parser.parse(raw_document)
                if parsed.markdown.strip():
                    parsed_documents.append(parsed)
            except Exception as exc:
                print(f"[Warning] Failed to parse '{raw_document.relative_path}': {exc}")

        return parsed_documents