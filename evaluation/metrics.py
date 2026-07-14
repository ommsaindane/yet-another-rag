from deepeval import evaluate

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from deepeval.test_case import LLMTestCase

from evaluation.ollama_model import OllamaDeepEvalModel


class EvaluationMetrics:
    def __init__(
        self,
        threshold: float = 0.7,
    ):
        judge = OllamaDeepEvalModel()

        self.metrics = [
            AnswerRelevancyMetric(
                model=judge,
                threshold=threshold,
            ),
            FaithfulnessMetric(
                model=judge,
                threshold=threshold,
            ),
            ContextualPrecisionMetric(
                model=judge,
                threshold=threshold,
            ),
            ContextualRecallMetric(
                model=judge,
                threshold=threshold,
            ),
        ]

    def evaluate(
        self,
        test_cases: list[LLMTestCase],
    ):
        return evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
        )

