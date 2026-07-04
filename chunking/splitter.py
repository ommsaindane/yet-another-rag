from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ):
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
