from test.ragtest import build_rag

from evaluation.dataset import EvaluationDataset
from evaluation.runner import EvaluationRunner
from evaluation.metrics import EvaluationMetrics
from dotenv import load_dotenv

load_dotenv()


def main():

    print("Building RAG pipeline...")

    graph = build_rag()

    print("Loading evaluation dataset...")

    dataset = EvaluationDataset(
        "data/eval/questions.json",
    )

    print("Generating test cases...")

    runner = EvaluationRunner(
        graph=graph,
        dataset=dataset,
    )

    test_cases = runner.run()

    print(f"Generated {len(test_cases)} test cases.")

    print("Running evaluation...")

    metrics = EvaluationMetrics(
        threshold=0.7,
    )

    results = metrics.evaluate(test_cases)

    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(results)
    print()


if __name__ == "__main__":
    main()
