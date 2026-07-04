from uuid import uuid4

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from ingestion.loader import RawDocument
from models.document import ParsedDocument


class DoclingParser:
    def __init__(self):

        pdf_options = PdfPipelineOptions()

        pdf_options.do_ocr = False
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
            id=str(uuid4()),
            markdown=doc.export_to_markdown(),
            metadata={
                "filename": document.filename,
                "extension": document.extension,
                "size": document.size,
                "path": str(document.path),
            },
            docling_document=doc,
        )
