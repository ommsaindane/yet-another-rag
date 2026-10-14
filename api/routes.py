import json
from collections.abc import Iterator
from time import perf_counter
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.graph import CRAGGraph
from agent.state import AgentState
from api.dependencies import graph_dependency
from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    SyncResponse,
)
from api.startup import initialize_graph

router = APIRouter()


def _extract_sources(state: AgentState) -> list[SourceResponse]:
    """
    Extracts deduplicated local or web sources depending on whether local context
    passed relevance grading.
    """
    sources: list[SourceResponse] = []

    if state["context_is_relevant"]:
        seen_local: set[tuple[str, str | None]] = set()
        for parent in state["relevant_parents"]:
            filename = parent.metadata.get("filename", parent.document_id)
            section = parent.metadata.get("section_title")
            key = (filename, section)
            if key in seen_local:
                continue
            seen_local.add(key)
            sources.append(
                SourceResponse(
                    source_type="local",
                    document=filename,
                    section=section,
                    score=parent.metadata.get("retrieval_score"),
                )
            )
    else:
        seen_urls: set[str] = set()
        for web_res in state["web_results"]:
            if not web_res.url or web_res.url in seen_urls:
                continue
            seen_urls.add(web_res.url)
            sources.append(
                SourceResponse(
                    source_type="web",
                    document=web_res.title,
                    url=web_res.url,
                )
            )

    return sources


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    graph: CRAGGraph = Depends(graph_dependency),
) -> QueryResponse:
    start = perf_counter()

    try:
        state = graph.invoke(request.query)
        latency = round(perf_counter() - start, 4)
        web_fallback = not state["context_is_relevant"]

        return QueryResponse(
            answer=state["answer"],
            sources=_extract_sources(state),
            web_fallback=web_fallback,
            rewritten_query=state.get("search_query") if web_fallback else None,
            latency=latency,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {exc}",
        ) from exc


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    graph: CRAGGraph = Depends(graph_dependency),
) -> StreamingResponse:
    """
    Streams the RAG response using Server-Sent Events (SSE).
    Emits:
      - `event: metadata` with sources, web_fallback, and rewritten_query
      - `event: token` for each generated text chunk
      - `event: done` with total latency
    """
    start = perf_counter()

    try:
        state, token_iterator = graph.stream(request.query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize stream: {exc}",
        ) from exc

    def _event_generator() -> Iterator[str]:
        web_fallback = not state["context_is_relevant"]
        sources = [s.model_dump() for s in _extract_sources(state)]

        metadata_payload = {
            "sources": sources,
            "web_fallback": web_fallback,
            "rewritten_query": state.get("search_query") if web_fallback else None,
        }
        yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"

        for token in token_iterator:
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        latency = round(perf_counter() - start, 4)
        yield f"event: done\ndata: {json.dumps({'latency': latency})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
    )


@router.post(
    "/index/sync",
    response_model=SyncResponse,
)
def sync_index(request: Request) -> SyncResponse:
    """
    Triggers an incremental index sync if documents in `data/raw` were added,
    modified, or removed.
    """
    try:
        initialize_graph(request.app)
        return SyncResponse(
            status="ok",
            message="Index synchronized and retrieval pipeline refreshed.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Index synchronization failed: {exc}",
        ) from exc