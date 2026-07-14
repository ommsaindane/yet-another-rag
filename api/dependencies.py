from agent.graph import CRAGGraph

from api.startup import get_graph


def graph_dependency() -> CRAGGraph:
    """
    FastAPI dependency that returns the shared CRAG graph instance.
    """
    return get_graph()

