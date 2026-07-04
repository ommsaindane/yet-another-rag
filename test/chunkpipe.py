from ingestion.pipeline import IngestionPipeline
from chunking.pipeline import ChunkingPipeline


def main():
    ingestion = IngestionPipeline("data/raw")
    chunking = ChunkingPipeline()

    parsed_documents = ingestion.run()

    print(f"Parsed {len(parsed_documents)} document(s).\n")

    parent_chunks, child_chunks = chunking.run(parsed_documents)

    print(f"Parent chunks : {len(parent_chunks)}")
    print(f"Child chunks  : {len(child_chunks)}")

    if parent_chunks:
        print("\n=== First Parent Chunk ===")
        print(parent_chunks[0].text[:500])

    if child_chunks:
        print("\n=== First Child Chunk ===")
        print(child_chunks[0].text[:300])


if __name__ == "__main__":
    main()
