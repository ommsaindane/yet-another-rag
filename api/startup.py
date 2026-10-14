from fastapi import FastAPI
from agent.builder import build_rag
from agent.graph import CRAGGraph


def initialize_graph(app: FastAPI) -> CRAGGraph:
    graph = build_rag()
    app.state.graph = graph
    return graph