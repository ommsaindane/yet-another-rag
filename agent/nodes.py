from pydantic import BaseModel, Field
from agent.state import AgentState
from llm.models import LlamaModel
from llm.prompts import PromptManager
from models.chunk import ParentChunk
from retrieval.pipeline import RetrievalPipeline
from web.search import DuckDuckGoSearch


class GradeResult(BaseModel):
    is_relevant: bool = Field(
        description="True if the chunk contains information relevant to answering the question."
    )


class AgentNodes:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        llm: LlamaModel,
        prompts: PromptManager,
        web_search: DuckDuckGoSearch,
    ):
        self.retrieval = retrieval
        self.llm = llm
        self.prompts = prompts
        self.search_engine = web_search

    def retrieve(
        self,
        state: AgentState,
    ) -> AgentState:
        state["retrieved_parents"] = self.retrieval.retrieve(
            query=state["query"],
        )
        return state

    def grade_context(
        self,
        state: AgentState,
    ) -> AgentState:
        retrieved = state.get("retrieved_parents", [])
        if not retrieved:
            state["relevant_parents"] = []
            state["context_is_relevant"] = False
            return state

        relevant_parents: list[ParentChunk] = []

        for chunk in retrieved:
            system_prompt, user_prompt = self.prompts.get(
                "grader",
                context=chunk.text,
                question=state["query"],
            )
            try:
                grade = self.llm.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=GradeResult,
                )
                is_rel = grade.is_relevant
            except Exception:
                # Fallback to substring check if structured output fails
                raw = (
                    self.llm.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.0,
                    )
                    .strip()
                    .upper()
                )
                is_rel = "YES" in raw or "TRUE" in raw

            if is_rel:
                relevant_parents.append(chunk)

        state["relevant_parents"] = relevant_parents
        state["context_is_relevant"] = len(relevant_parents) > 0
        return state

    def rewrite_query(
        self,
        state: AgentState,
    ) -> AgentState:
        from pydantic import BaseModel
        
        class RewrittenQuery(BaseModel):
            search_query: str

        system_prompt, user_prompt = self.prompts.get(
            "query_rewriter",
            question=state["query"],
        )
        
        try:
            result = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=RewrittenQuery,
            )
            rewritten = result.search_query.strip()
        except Exception:
            # Fallback to the original query if the model fails the JSON schema
            rewritten = state["query"]

        state["search_query"] = rewritten if rewritten else state["query"]
        return state

    def web_search(
        self,
        state: AgentState,
    ) -> AgentState:
        search_query = state.get("search_query") or state["query"]
        state["web_results"] = self.search_engine.search(search_query)
        return state

    def build_context(
        self,
        state: AgentState,
    ) -> AgentState:
        context_parts: list[str] = []

        if state["context_is_relevant"]:
            for chunk in state["relevant_parents"]:
                filename = chunk.metadata.get("filename", chunk.document_id)
                section = chunk.metadata.get("section_title", "")
                header = (
                    f""
                    if section
                    else f""
                )
                context_parts.append(f"{header}\n{chunk.text}")
        else:
            for result in state["web_results"]:
                header = f""
                context_parts.append(f"{header}\n{result.snippet}")

        state["retrieved_contexts"] = context_parts
        state["context"] = "\n\n".join(context_parts)
        return state

    def generate(
        self,
        state: AgentState,
    ) -> AgentState:
        if not state["context"].strip():
            state["answer"] = "I don't have enough information to answer that."
            return state

        system_prompt, user_prompt = self.prompts.get(
            "generator",
            context=state["context"],
            question=state["query"],
        )
        state["answer"] = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return state