from typing import Literal
from agent.state import AgentState


class AgentRouter:
    @staticmethod
    def route_after_grading(
        state: AgentState,
    ) -> Literal["build_context", "rewrite_query"]:
        if state["context_is_relevant"]:
            return "build_context"
        return "rewrite_query"