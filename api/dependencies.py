from fastapi import Request
from agent.graph import CRAGGraph


def graph_dependency(request: Request) -> CRAGGraph:
    """
    FastAPI dependency that returns the shared CRAG graph instance from app.state.
    """
    graph: CRAGGraph | None = getattr(request.app.state, "graph", None)
    if graph is None:
        raise RuntimeError("CRAG graph has not been initialized.")
    return graph