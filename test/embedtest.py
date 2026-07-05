from embedding.embedder import BGEEmbedder
from embedding.pipeline import EmbeddingPipeline
from chunking.pipeline import ChunkingPipeline
from ingestion.pipeline import IngestionPipeline
from vectordb.chroma import ChromaVectorStore


def main() -> None:

    # ----------------------------
    # Ingestion
    # ----------------------------

    ingestion = IngestionPipeline("data/raw")

    parsed_documents = ingestion.run()

    print(f"Loaded documents : {len(parsed_documents)}")

    # ----------------------------
    # Chunking
    # ----------------------------

    chunking = ChunkingPipeline()

    parent_chunks, child_chunks = chunking.run(parsed_documents)

    print(f"Parent chunks    : {len(parent_chunks)}")
    print(f"Child chunks     : {len(child_chunks)}")

    # ----------------------------
    # Embedding + Indexing
    # ----------------------------

    embedding = EmbeddingPipeline(
        persist_directory="data/chroma",
        collection_name="test_collection",
    )

    embedding.run(child_chunks)

    db = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="test_collection",
    )

    print(f"Indexed vectors  : {db.count()}")

    # ----------------------------
    # Retrieval Test
    # ----------------------------

    embedder = BGEEmbedder()

    query = "Summarize the main topic of this document."

    query_embedding = embedder.model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = db.search(query_embedding, k=3)

    print("\n========== Retrieval Test ==========\n")

    print("Query:")
    print(query)

    print("\nTop Match:\n")

    print(results["documents"][0][0])

    print("\nMetadata:\n")

    print(results["metadatas"][0][0])

    print("\nDistance:")

    print(results["distances"][0][0])


if __name__ == "__main__":
    main()
