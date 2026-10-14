from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from evaluation.ollama_model import OllamaDeepEvalModel


class EvaluationMetrics:
    def __init__(
        self,
        threshold: float = 0.7,
        fast_mode: bool = False,
    ):
        judge = OllamaDeepEvalModel()

        # Core generation metrics
        self.metrics = [
            AnswerRelevancyMetric(
                model=judge,
                threshold=threshold,
                async_mode=False,
            ),
            FaithfulnessMetric(
                model=judge,
                threshold=threshold,
                async_mode=False,
            ),
        ]

        # Full retrieval + generation triad when not in fast_mode
        if not fast_mode:
            self.metrics.extend(
                [
                    ContextualPrecisionMetric(
                        model=judge,
                        threshold=threshold,
                        async_mode=False,
                    ),
                    ContextualRecallMetric(
                        model=judge,
                        threshold=threshold,
                        async_mode=False,
                    ),
                ]
            )

    def evaluate(
        self,
        test_cases: list[LLMTestCase],
    ):
        return evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
            async_config=AsyncConfig(
                run_async=False,
                max_concurrent=1,
            ),
            error_config=ErrorConfig(
                ignore_errors=True,
            ),
        )