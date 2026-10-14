from typing import TypedDict
from models.chunk import ParentChunk
from models.web import WebResult


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """

    # Input
    query: str
    search_query: str

    # Retrieval & Grading
    retrieved_parents: list[ParentChunk]
    relevant_parents: list[ParentChunk]
    context_is_relevant: bool

    # Web fallback
    web_results: list[WebResult]

    # Final generation context
    context: str
    retrieved_contexts: list[str]

    # Output
    answer: str