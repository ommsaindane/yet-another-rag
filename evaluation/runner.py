from deepeval.test_case import LLMTestCase

from evaluation.dataset import EvaluationDataset

from agent.graph import CRAGGraph


class EvaluationRunner:
    def __init__(
        self,
        graph: CRAGGraph,
        dataset: EvaluationDataset,
    ):
        self.graph = graph
        self.dataset = dataset

    def run(self) -> list[LLMTestCase]:

        test_cases: list[LLMTestCase] = []

        for sample in self.dataset.load():
            result = self.graph.invoke(
                sample["question"],
            )

            test_cases.append(
                LLMTestCase(
                    input=sample["question"],
                    actual_output=result["answer"],
                    expected_output=sample["expected_answer"],
                    retrieval_context=result["retrieved_contexts"],
                )
            )

        return test_cases
