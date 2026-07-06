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

from llm.models import LlamaModel
from llm.prompts import PromptManager

from web.search import DuckDuckGoSearch

from agent.graph import CRAGGraph


def build_rag():

    print("Ingesting documents...")

    ingestion = IngestionPipeline("data/raw")
    parsed_documents = ingestion.run()

    print(f"Loaded {len(parsed_documents)} documents.")
    print()

    print("Chunking...")

    chunking = ChunkingPipeline()

    parent_chunks, child_chunks = chunking.run(parsed_documents)

    print(f"{len(parent_chunks)} parent chunks")
    print(f"{len(child_chunks)} child chunks")
    print()

    print("Embedding...")

    embedding = EmbeddingPipeline(
        persist_directory="data/chroma",
        collection_name="papers",
    )

    embedding.run(
        child_chunks,
        overwrite=True,
    )

    print("Building retrieval...")

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

    parent_store = ParentStore()
    parent_store.add(parent_chunks)

    retrieval = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        parent_store=parent_store,
    )

    print("Loading LLM...")

    llm = LlamaModel()
    prompts = PromptManager()
    web_search = DuckDuckGoSearch()

    graph = CRAGGraph(
        retrieval=retrieval,
        llm=llm,
        prompts=prompts,
        web_search=web_search,
    )

    return graph


def main():

    graph = build_rag()

    print()

    while True:
        query = input("Question (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = graph.invoke(query)

        print()
        print("=" * 80)
        print(result["answer"])
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()

