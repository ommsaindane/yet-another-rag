from pathlib import Path

from ingestion.loader import DocumentLoader
from ingestion.parser import DoclingParser, ParsedDocument


class IngestionPipeline:
    def __init__(self, data_dir: str | Path):
        self.loader = DocumentLoader(data_dir)
        self.parser = DoclingParser()

    def run(self) -> list[ParsedDocument]:

        parsed_documents: list[ParsedDocument] = []

        for raw_document in self.loader.load():
            parsed_documents.append(self.parser.parse(raw_document))

        return parsed_documents
