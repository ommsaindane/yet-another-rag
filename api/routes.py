from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException

from agent.graph import CRAGGraph

from api.dependencies import graph_dependency
from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

router = APIRouter()


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

        latency = perf_counter() - start

        sources: list[SourceResponse] = []
        seen: set[str] = set()

        for parent in state["retrieved_parents"]:
            filename = parent.metadata.get("filename", parent.document_id)

            if filename in seen:
                continue

            seen.add(filename)

            sources.append(
                SourceResponse(
                    document=filename,
                )
            )

        return QueryResponse(
            answer=state["answer"],
            sources=sources,
            web_fallback=not state["context_is_relevant"],
            latency=latency,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
