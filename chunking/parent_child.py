from chunking.splitter import TextSplitter
from models.chunk import ChildChunk, ParentChunk
from models.document import ParsedDocument


class ParentChildChunker:
    """
    Generates deterministic, structure-aware hierarchical parent-child chunks
    from a parsed document.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = TextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        self.child_splitter = TextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )

    def chunk(
        self,
        document: ParsedDocument,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        parent_sections = self.parent_splitter.split_with_sections(document.markdown)

        for p_idx, section in enumerate(parent_sections):
            parent_id = f"{document.id}_p{p_idx}"

            parent_meta = {
                **document.metadata,
                "section_title": section.section_title,
                "parent_index": p_idx,
            }

            parent_chunk = ParentChunk(
                id=parent_id,
                document_id=document.id,
                text=section.text,
                metadata=parent_meta,
            )
            parent_chunks.append(parent_chunk)

            child_texts = self.child_splitter.split(section.text)

            for c_idx, child_text in enumerate(child_texts):
                child_id = f"{parent_id}_c{c_idx}"

                # Prepend section context if the child chunk doesn't already start with a header
                enriched_child_text = child_text
                if (
                    section.section_title
                    and section.section_title != "Introduction"
                    and not child_text.lstrip().startswith("#")
                ):
                    enriched_child_text = (
                        f"[Section: {section.section_title}]\n{child_text}"
                    )

                child_meta = {
                    **parent_meta,
                    "child_index": c_idx,
                }

                child_chunk = ChildChunk(
                    id=child_id,
                    parent_id=parent_id,
                    document_id=document.id,
                    text=enriched_child_text,
                    metadata=child_meta,
                )
                child_chunks.append(child_chunk)

        return parent_chunks, child_chunks