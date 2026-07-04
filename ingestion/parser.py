from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import DoclingDocument

from ingestion.loader import RawDocument


@dataclass(slots=True)
class ParsedDocument:
    markdown: str
    metadata: dict
    docling_document: DoclingDocument


class DoclingParser:
    def __init__(self):

        pdf_options = PdfPipelineOptions()

        pdf_options.do_ocr = True
        pdf_options.ocr_options = EasyOcrOptions()

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )

    def parse(self, document: RawDocument) -> ParsedDocument:

        result = self.converter.convert(str(document.path))

        doc = result.document

        return ParsedDocument(
            markdown=doc.export_to_markdown(),
            metadata={
                "filename": document.filename,
                "extension": document.extension,
                "size": document.size,
                "path": str(document.path),
            },
            docling_document=doc,
        )
