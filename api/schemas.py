from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )


class SourceResponse(BaseModel):
    source_type: Literal["local", "web"] = Field(
        description="Whether this source came from local documents or web search."
    )
    document: str = Field(
        description="Document filename (if local) or webpage title (if web)."
    )
    section: str | None = Field(
        default=None,
        description="Markdown section breadcrumb for local document chunks.",
    )
    url: str | None = Field(
        default=None,
        description="URL for web search fallback results.",
    )
    score: float | None = Field(
        default=None,
        description="Normalized reranker relevance score for local chunks.",
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer.")
    sources: list[SourceResponse] = Field(
        default_factory=list,
        description="Verified sources used to generate the answer.",
    )
    web_fallback: bool = Field(
        description="Whether the agent fell back to web search."
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Optimized search query used if web fallback was triggered.",
    )
    latency: float = Field(description="End-to-end request latency in seconds.")


class SyncResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str