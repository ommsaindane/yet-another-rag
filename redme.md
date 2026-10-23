# Yet Another RAG

A Corrective RAG (CRAG) pipeline that performs hybrid retrieval, reranking, context grading, and automatic web fallback to answer questions over your document collection.

🌟 **Live Demo**: You can try the exact same project hosted on Hugging Face Spaces with a Gradio UI here: [Yet Another RAG on HF Spaces](https://huggingface.co/spaces/omsaindane/yet-another-rag)

# Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)

# Architecture
<img width="1319" height="1260" alt="Image" src="https://github.com/user-attachments/assets/d170bc45-523b-451c-9298-8bf88de1c09b" />

# Output Example
<img width="1902" height="965" alt="Image" src="https://github.com/user-attachments/assets/b9647d60-82b9-48de-a8fe-4621c457d1fa" />

# Features
- Local-first document question answering
- Corrective RAG (CRAG) workflow using LangGraph
- Automatic document ingestion pipeline
- PDF parsing using Docling
- Hierarchical Parent-Child Chunking
- Hybrid Retrieval (Dense + BM25)
- Reciprocal Rank Fusion (RRF)
- BGE Reranker
- Local Llama 3.2 Instruct inference
- Automatic web search fallback when retrieved context is insufficient
- FastAPI REST API
- Automatic document change detection to avoid unnecessary re-indexing
- Persistent Chroma vector database
- Evaluation pipeline using DeepEval
- **Interactive Streaming CLI Chat**
- **Local Streamlit Console UI & Hosted Gradio UI**

# Tech Stack
| Component | Technology |
|-----------|------------|
| Parsing | Docling |
| OCR | EasyOCR |
| Chunking | Parent-Child Hierarchical Chunking |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 |
| Hybrid Fusion | Reciprocal Rank Fusion |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Llama 3.2 3B Instruct |
| Agent Framework | LangGraph |
| Web Search | DuckDuckGo |
| API | FastAPI |
| UI | Streamlit / Gradio |
| Evaluation | DeepEval |

# Getting Started
## Prerequisites
- Python 3.11+
- Ollama
- Git

## Installation
Clone the repository
```bash
git clone https://github.com/ommsaindane/yet-another-rag.git

cd yet-another-rag
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install .
```

Pull the Llama model

```bash
ollama pull llama3.2:3b-instruct
```

Place your PDFs inside

```text
data/raw/
```

# Usage

The project provides a unified CLI via `main.py` to interact with the pipeline.

## API

Start the server

```bash
python main.py serve --reload

```

Swagger UI

```text
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

```

Example request

```http
POST /query

```

```json
{
    "query": "What is Corrective RAG?"
}

```

Example response

```json
{
    "answer": "...",
    "sources": [
        {
            "document": "paper.pdf"
        }
    ],
    "web_fallback": false,
    "latency": 2.84
}

```

## UI

Launch the local Streamlit console UI

```bash
python main.py ui

```

*Note: A hosted version of this project using a Gradio UI is also available on [Hugging Face Spaces](https://huggingface.co/spaces/omsaindane/yet-another-rag).*

## CLI Chat

Start an interactive streaming CLI chat session

```bash
python main.py chat

```

## Sync

Incrementally sync `data/raw` with ChromaDB and JSON stores

```bash
python main.py sync

```

## Evaluation

Run the DeepEval benchmark suite

```bash
python main.py eval

```

*(Add `--fast` to run only Faithfulness and AnswerRelevancy for faster iteration)*

# Design Decisions

* Parent-child hierarchical chunking improves retrieval quality while providing larger context windows to the LLM.
* Character-based chunking is used instead of page-based chunking, allowing chunks to span page boundaries. Because of this, page numbers are intentionally not exposed by the API.
* Only child chunks are embedded and indexed, while parent chunks are stored separately and retrieved after reranking.
* A manifest tracks document changes, preventing unnecessary parsing, chunking, and embedding on subsequent runs.

# Roadmap

- [x] Local RAG pipeline
- [x] Hybrid retrieval
- [x] Parent-child chunking
- [x] LangGraph CRAG agent
- [x] FastAPI REST API
- [x] Automatic document change detection
- [x] Evaluation pipeline
- [x] Streaming responses
- [ ] Docker support
- [ ] LangSmith integration
- [ ] Multi-document collections
- [ ] Citation highlighting
- [ ] Authentication