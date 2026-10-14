from dataclasses import dataclass
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


_HEADER_REGEX = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


@dataclass(slots=True)
class SectionText:
    text: str
    section_title: str


class TextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def split_with_sections(self, markdown: str) -> list[SectionText]:
        """
        Splits Markdown while tracking the active heading hierarchy (# > ## > ###)
        so each resulting chunk retains its section breadcrumb.
        """
        lines = markdown.splitlines()
        header_stack: dict[int, str] = {}
        blocks: list[SectionText] = []
        current_lines: list[str] = []
        current_breadcrumb = "Introduction"

        def _flush_block() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append(
                    SectionText(
                        text=content,
                        section_title=current_breadcrumb,
                    )
                )

        for line in lines:
            match = _HEADER_REGEX.match(line.strip())
            if match:
                _flush_block()
                current_lines = [line]
                level = len(match.group(1))
                title = match.group(2).strip()

                # Update current header level and clear deeper sub-headers
                header_stack[level] = title
                for deeper in [k for k in header_stack if k > level]:
                    del header_stack[deeper]

                current_breadcrumb = " > ".join(
                    header_stack[k] for k in sorted(header_stack)
                )
            else:
                current_lines.append(line)

        _flush_block()

        # Merge small adjacent sections or split oversized sections to respect chunk_size
        final_chunks: list[SectionText] = []
        buffer_text = ""
        buffer_section = "Introduction"

        for block in blocks:
            candidate = f"{buffer_text}\n\n{block.text}".strip() if buffer_text else block.text
            if len(candidate) <= self.chunk_size:
                if not buffer_text:
                    buffer_section = block.section_title
                buffer_text = candidate
            else:
                if buffer_text:
                    final_chunks.append(
                        SectionText(text=buffer_text, section_title=buffer_section)
                    )
                    buffer_text = ""

                if len(block.text) <= self.chunk_size:
                    buffer_text = block.text
                    buffer_section = block.section_title
                else:
                    for sub_chunk in self.split(block.text):
                        final_chunks.append(
                            SectionText(
                                text=sub_chunk,
                                section_title=block.section_title,
                            )
                        )

        if buffer_text:
            final_chunks.append(
                SectionText(text=buffer_text, section_title=buffer_section)
            )

        return final_chunks