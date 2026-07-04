from ingestion.loader import DocumentLoader
from ingestion.parser import DoclingParser


def main():
    loader = DocumentLoader("data/raw")
    parser = DoclingParser()

    documents = loader.load()

    if not documents:
        print("No documents found.")
        return

    for document in documents:
        print(f"\nParsing: {document.filename}")

        parsed = parser.parse(document)

        print("-" * 80)
        print(parsed.text[:1000])  # First 1000 characters
        print("-" * 80)
        print(parsed.metadata)


if __name__ == "__main__":
    main()
