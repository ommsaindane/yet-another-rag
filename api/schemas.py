from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The user's question.",
    )


class SourceResponse(BaseModel):
    document: str = Field(description="Document name or path.")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer.")
    sources: list[SourceResponse] = Field(
        default_factory=list,
        description="Documents used to generate the answer.",
    )
    web_fallback: bool = Field(description="Whether the answer required web search.")
    latency: float = Field(description="End-to-end request latency in seconds.")


class HealthResponse(BaseModel):
    status: str
