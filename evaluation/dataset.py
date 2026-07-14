from pathlib import Path
import json


class EvaluationDataset:
    def __init__(
        self,
        dataset_path: str | Path = "data/eval/questions.json",
    ):
        self.dataset_path = Path(dataset_path)

    def load(self) -> list[dict]:

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
        ) as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Evaluation dataset must be a list.")

        required_fields = {
            "id",
            "question",
            "expected_answer",
        }

        for sample in dataset:
            missing = required_fields - sample.keys()

            if missing:
                raise ValueError(
                    f"Sample {sample.get('id', '?')} is missing fields: {missing}"
                )

        return dataset
