from agent.builder import build_rag
from agent.graph import CRAGGraph

_graph: CRAGGraph | None = None


def initialize_graph() -> None:
    global _graph

    if _graph is None:
        _graph = build_rag()


def get_graph() -> CRAGGraph:
    if _graph is None:
        raise RuntimeError("CRAG graph has not been initialized.")

    return _graph
