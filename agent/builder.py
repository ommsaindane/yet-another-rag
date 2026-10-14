from pathlib import Path

from agent.graph import CRAGGraph
from chunking.pipeline import ChunkingPipeline
from config.settings import get_settings
from embedding.embedder import BGEEmbedder
from embedding.pipeline import EmbeddingPipeline
from ingestion.loader import DocumentLoader
from ingestion.pipeline import IngestionPipeline
from llm.models import LlamaModel
from llm.prompts import PromptManager
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import BGEReranker
from retrieval.vector import VectorRetriever
from storage.child_store import ChildStore
from storage.manifest import Manifest
from storage.parent_store import ParentStore
from vectordb.chroma import ChromaVectorStore
from web.search import DuckDuckGoSearch


def build_rag(
    raw_data_dir: str | Path | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
) -> CRAGGraph:
    settings = get_settings()
    raw_dir = Path(raw_data_dir or settings.raw_data_dir)
    chroma_dir = Path(persist_directory or settings.chroma_dir)
    col_name = collection_name or settings.collection_name

    manifest = Manifest(
        raw_directory=raw_dir,
        manifest_path=settings.manifest_path,
        supported_extensions=DocumentLoader.SUPPORTED_EXTENSIONS,
    )
    parent_store = ParentStore(store_path=settings.parent_store_path)
    child_store = ChildStore(store_path=settings.child_store_path)

    vector_store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name=col_name,
    )
    embedder = BGEEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    # Load existing stores if present on disk
    stores_exist = parent_store.exists() and child_store.exists()
    if stores_exist:
        parent_store.load()
        child_store.load()

    diff = manifest.get_diff()

    # Force a full rebuild if index files are missing despite manifest existing
    if not stores_exist and not diff.has_changes and raw_dir.exists():
        loader = DocumentLoader(raw_dir)
        diff.added = [doc.path for doc in loader.load()]
        vector_store.delete()

    if diff.has_changes:
        print(
            f"Index sync triggered — Added: {len(diff.added)}, "
            f"Modified: {len(diff.modified)}, Removed: {len(diff.removed)}"
        )

        # 1. Purge stale chunks & vectors for removed or modified documents
        modified_rel_paths = [manifest.get_relative_path(p) for p in diff.modified]
        for rel_path in diff.removed + modified_rel_paths:
            doc_id = Manifest.compute_document_id(rel_path)
            parent_store.delete_by_document_id(doc_id)
            child_store.delete_by_document_id(doc_id)
            vector_store.delete_by_document_id(doc_id)

        # 2. Ingest, chunk, and embed ONLY added or modified files
        files_to_process = diff.files_to_ingest
        if files_to_process:
            print(f"Ingesting {len(files_to_process)} changed document(s)...")
            ingestion = IngestionPipeline(raw_dir)
            parsed_documents = ingestion.run(file_paths=files_to_process)

            print("Chunking updated documents...")
            chunking = ChunkingPipeline(
                parent_chunk_size=settings.parent_chunk_size,
                parent_chunk_overlap=settings.parent_chunk_overlap,
                child_chunk_size=settings.child_chunk_size,
                child_chunk_overlap=settings.child_chunk_overlap,
            )
            new_parents, new_children = chunking.run(parsed_documents)
            print(
                f"Generated {len(new_parents)} parent chunks and "
                f"{len(new_children)} child chunks."
            )

            print("Embedding and indexing new chunks...")
            embedding = EmbeddingPipeline(
                embedder=embedder,
                vector_store=vector_store,
            )
            embedding.run(new_children, overwrite=False)

            parent_store.add(new_parents)
            child_store.add(new_children)

        parent_store.save()
        child_store.save()
        manifest.save()
    else:
        print("No document changes detected. Loaded existing indexes.")

    child_chunks = child_store.get()

    print("Building retrieval pipeline...")
    vector = VectorRetriever(
        vector_store=vector_store,
        embedder=embedder,
    )

    bm25 = BM25Retriever()
    if child_chunks:
        bm25.index(child_chunks)

    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        rrf_k=settings.rrf_k,
    )

    reranker = BGEReranker(model_name=settings.reranker_model)

    retrieval = RetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        parent_store=parent_store,
    )

    print("Loading LLM...")
    llm = LlamaModel(
        model_name=settings.llm_model,
        host=settings.ollama_host,
    )
    prompts = PromptManager(prompt_file=settings.prompts_path)
    web_search = DuckDuckGoSearch(max_results=settings.web_search_max_results)

    return CRAGGraph(
        retrieval=retrieval,
        llm=llm,
        prompts=prompts,
        web_search=web_search,
    )