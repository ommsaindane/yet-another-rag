from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----------------------------
    # Directories & File Paths
    # ----------------------------
    raw_data_dir: Path = Path("data/raw")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "papers"
    parent_store_path: Path = Path("data/indexes/parent_store.json")
    child_store_path: Path = Path("data/indexes/child_store.json")
    manifest_path: Path = Path("data/manifest.json")
    prompts_path: Path = Path("config/prompts.yaml")
    eval_dataset_path: Path = Path("data/eval/questions.json")

    # ----------------------------
    # Models & Hosts
    # ----------------------------
    llm_model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"

    # ----------------------------
    # Chunking Hyperparameters
    # ----------------------------
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # ----------------------------
    # Retrieval & Web Search
    # ----------------------------
    retrieve_k: int = 20
    rerank_k: int = 5
    rrf_k: int = 60
    web_search_max_results: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()