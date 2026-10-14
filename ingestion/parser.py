from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from models.document import ParsedDocument, RawDocument
from storage.manifest import Manifest


class DoclingParser:
    def __init__(self, do_ocr: bool = False):
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = do_ocr
        pdf_options.ocr_options = EasyOcrOptions()

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )

    def parse(self, document: RawDocument) -> ParsedDocument:
        doc_id = Manifest.compute_document_id(document.relative_path)

        if document.extension in {".md", ".txt"}:
            markdown = document.path.read_text(encoding="utf-8", errors="replace")
            page_count = 1
        else:
            result = self.converter.convert(str(document.path))
            doc = result.document
            markdown = doc.export_to_markdown()
            page_count = len(getattr(doc, "pages", {})) or 1

        return ParsedDocument(
            id=doc_id,
            markdown=markdown,
            metadata={
                "filename": document.filename,
                "relative_path": document.relative_path,
                "extension": document.extension,
                "size": document.size,
                "page_count": page_count,
            },
        )