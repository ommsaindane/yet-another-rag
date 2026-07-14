from typing import TypedDict

from models.chunk import ParentChunk
from models.web import WebResult


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """

    #
    # Input
    #

    query: str

    #
    # Retrieval
    #

    retrieved_parents: list[ParentChunk]

    #
    # Context grading
    #

    context_is_relevant: bool

    #
    # Web search
    #

    web_results: list[WebResult]

    #
    # Final generation context
    #

    context: str
    retrieved_contexts: list[str]
    #
    # Output
    #

    answer: str
