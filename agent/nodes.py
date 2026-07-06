from agent.state import AgentState

from llm.models import LlamaModel
from llm.prompts import PromptManager
from retrieval.pipeline import RetrievalPipeline


class AgentNodes:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        llm: LlamaModel,
        prompts: PromptManager,
    ):
        self.retrieval = retrieval
        self.llm = llm
        self.prompts = prompts

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

        context = "\n\n".join(chunk.text for chunk in state["retrieved_parents"])

        system_prompt, user_prompt = self.prompts.get(
            "grader",
            context=context,
            question=state["query"],
        )

        response = (
            self.llm.generate(
                system_prompt,
                user_prompt,
            )
            .strip()
            .upper()
        )

        if response == "YES":
            state["context_is_relevant"] = True
        elif response == "NO":
            state["context_is_relevant"] = False
        else:
            raise ValueError(f"Invalid grader output: {response}")

        return state

    def web_search(
        self,
        state: AgentState,
    ) -> AgentState:

        state["web_results"] = []

        return state

    def build_context(
        self,
        state: AgentState,
    ) -> AgentState:

        context_parts: list[str] = []

        if state["context_is_relevant"]:
            context_parts.extend(chunk.text for chunk in state["retrieved_parents"])

            context_parts.extend(
                f"{result.title}\n{result.snippet}" for result in state["web_results"]
            )
            state["context"] = "\n\n".join(context_parts)

        return state

    def generate(
        self,
        state: AgentState,
    ) -> AgentState:

        system_prompt, user_prompt = self.prompts.get(
            "generator",
            context=state["context"],
            question=state["query"],
        )

        state["answer"] = self.llm.generate(
            system_prompt,
            user_prompt,
        )

        return state
