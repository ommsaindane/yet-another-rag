from collections.abc import Iterator
from typing import Any
from langgraph.graph import END, START, StateGraph
from agent.nodes import AgentNodes
from agent.router import AgentRouter
from agent.state import AgentState
from llm.models import LlamaModel
from llm.prompts import PromptManager
from retrieval.pipeline import RetrievalPipeline
from web.search import DuckDuckGoSearch


class CRAGGraph:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        llm: LlamaModel,
        prompts: PromptManager,
        web_search: DuckDuckGoSearch,
    ):
        self.nodes = AgentNodes(
            retrieval=retrieval,
            llm=llm,
            prompts=prompts,
            web_search=web_search,
        )
        self.llm = llm
        self.prompts = prompts

        builder = StateGraph(AgentState)

        # Nodes
        builder.add_node("retrieve", self.nodes.retrieve)
        builder.add_node("grade_context", self.nodes.grade_context)
        builder.add_node("rewrite_query", self.nodes.rewrite_query)
        builder.add_node("web_search", self.nodes.web_search)
        builder.add_node("build_context", self.nodes.build_context)
        builder.add_node("generate", self.nodes.generate)

        # Edges
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_context")
        builder.add_conditional_edges(
            "grade_context",
            AgentRouter.route_after_grading,
            {
                "build_context": "build_context",
                "rewrite_query": "rewrite_query",
            },
        )
        builder.add_edge("rewrite_query", "web_search")
        builder.add_edge("web_search", "build_context")
        builder.add_edge("build_context", "generate")
        builder.add_edge("generate", END)

        self.graph = builder.compile()

    @staticmethod
    def _initial_state(query: str) -> AgentState:
        return {
            "query": query,
            "search_query": query,
            "retrieved_parents": [],
            "relevant_parents": [],
            "context_is_relevant": False,
            "web_results": [],
            "context": "",
            "retrieved_contexts": [],
            "answer": "",
        }

    def invoke(
        self,
        query: str,
    ) -> AgentState:
        return self.graph.invoke(self._initial_state(query))

    def prepare_context(
        self,
        query: str,
    ) -> AgentState:
        """
        Executes retrieval, grading, and optional web fallback up to build_context,
        returning the populated AgentState ready for streaming generation.
        """
        state = self._initial_state(query)
        state = self.nodes.retrieve(state)
        state = self.nodes.grade_context(state)
        if not state["context_is_relevant"]:
            state = self.nodes.rewrite_query(state)
            state = self.nodes.web_search(state)
        state = self.nodes.build_context(state)
        return state

    def stream(
        self,
        query: str,
    ) -> tuple[AgentState, Iterator[str]]:
        """
        Prepares the CRAG context and returns (state, token_iterator) for SSE streaming.
        """
        state = self.prepare_context(query)

        if not state["context"].strip():
            def _empty_gen() -> Iterator[str]:
                yield "I don't have enough information to answer that."
            return state, _empty_gen()

        system_prompt, user_prompt = self.prompts.get(
            "generator",
            context=state["context"],
            question=state["query"],
        )
        token_stream = self.llm.stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return state, token_stream