from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.router import AgentRouter
from agent.state import AgentState

from llm.models import LlamaModel
from llm.prompts import PromptManager

from retrieval.pipeline import RetrievalPipeline


class CRAGGraph:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        llm: LlamaModel,
        prompts: PromptManager,
    ):
        nodes = AgentNodes(
            retrieval=retrieval,
            llm=llm,
            prompts=prompts,
        )

        builder = StateGraph(AgentState)

        #
        # Nodes
        #

        builder.add_node("retrieve", nodes.retrieve)

        builder.add_node("grade_context", nodes.grade_context)

        builder.add_node("web_search", nodes.web_search)

        builder.add_node("build_context", nodes.build_context)

        builder.add_node("generate", nodes.generate)

        #
        # Edges
        #

        builder.add_edge(
            START,
            "retrieve",
        )

        builder.add_edge(
            "retrieve",
            "grade_context",
        )

        builder.add_conditional_edges(
            "grade_context",
            AgentRouter.route_after_grading,
            {
                "build_context": "build_context",
                "web_search": "web_search",
            },
        )

        builder.add_edge(
            "web_search",
            "build_context",
        )

        builder.add_edge(
            "build_context",
            "generate",
        )

        builder.add_edge(
            "generate",
            END,
        )

        self.graph = builder.compile()

    def invoke(
        self,
        query: str,
    ) -> AgentState:

        return self.graph.invoke(
            {
                "query": query,
                "retrieved_parents": [],
                "context_is_relevant": False,
                "web_results": [],
                "context": "",
                "answer": "",
            }
        )
