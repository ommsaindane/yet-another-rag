from ingestion.pipeline import IngestionPipeline
from chunking.pipeline import ChunkingPipeline
from embedding.pipeline import EmbeddingPipeline

from embedding.embedder import BGEEmbedder
from vectordb.chroma import ChromaVectorStore

from retrieval.vector import VectorRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import BGEReranker
from retrieval.parent_store import ParentStore
from retrieval.pipeline import RetrievalPipeline


def main():

    print("Ingesting documents...")

    ingestion = IngestionPipeline("data/raw")
    parsed_documents = ingestion.run()

    print(f"Loaded {len(parsed_documents)} documents.")

    print("Chunking...")

    chunking = ChunkingPipeline()
    parent_chunks, child_chunks = chunking.run(parsed_documents)

    print(f"{len(parent_chunks)} parent chunks")
    print(f"{len(child_chunks)} child chunks")

    print("Embedding and indexing...")

    embedding = EmbeddingPipeline(
        persist_directory="data/chroma",
        collection_name="papers",
    )

    embedding.run(child_chunks)

    print("Building retrieval pipeline...")

    parent_store = ParentStore()
    parent_store.add(parent_chunks)

    vector_store = ChromaVectorStore(
        persist_directory="data/chroma",
        collection_name="papers",
    )

    embedder = BGEEmbedder()

    vector = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    bm25 = BM25Retriever()
    bm25.index(child_chunks)

    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    reranker = BGEReranker()

    retrieval = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        parent_store=parent_store,
    )

    print()

    while True:
        query = input("Query (or 'exit'): ")

        if query.lower() == "exit":
            break

        parents = retrieval.retrieve(
            query=query,
            retrieve_k=20,
            rerank_k=5,
        )

        print()

        for i, parent in enumerate(parents, start=1):
            print("=" * 80)
            print(f"[Parent {i}]")
            print(f"Parent ID : {parent.id}")
            print()

            preview = parent.text

            if len(preview) > 1200:
                preview = preview[:1200] + "..."

            print(preview)
            print()

        print()


if __name__ == "__main__":
    main()
