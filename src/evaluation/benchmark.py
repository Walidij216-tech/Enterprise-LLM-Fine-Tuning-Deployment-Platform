import json
import csv

from pathlib import Path
from src.config.evaluation_config import EvaluationConfig

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsCalculator


class Benchmark:

    """
    Run and manage the model evaluation benchmark.
    """

    def __init__(self):

        self.test_dataset_path = (
            EvaluationConfig.TEST_DATASET_PATH
        )

        self.output_dir = EvaluationConfig.OUTPUT_DIR

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.evaluator = ModelEvaluator()
        self.metrics_calculator = MetricsCalculator()


    def save_metrics(self, metrics):

        metrics_path = self.output_dir / "metrics.json"

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"Metrics saved to: {metrics_path}")

    def save_predictions(
        self,
        predictions,
        test_dataset
    ):

        predictions_path = (
            self.output_dir / "predictions.csv"
        )

        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "example_id",
                    "input",
                    "expected_output",
                    "predicted_output",
                    "json_valid",
                    "error"
                ]
            )

            writer.writeheader()

            for prediction, example in zip(
                predictions,
                test_dataset
            ):

                writer.writerow({
                    "example_id": prediction["example_id"],
                    "input": example["input"],
                    "expected_output": json.dumps(
                        example["output"],
                        ensure_ascii=False
                    ),
                    "predicted_output": json.dumps(
                        prediction["output"],
                        ensure_ascii=False
                    ),
                    "json_valid": prediction["json_valid"],
                    "error": prediction["error"]
                })

        print(
            f"Predictions saved to: {predictions_path}"
        )

    @staticmethod
    def display_metrics(metrics):

        print("\n" + "=" * 55)
        print("MODEL EVALUATION RESULTS")
        print("=" * 55)

        print(
            f"Field Accuracy:        "
            f"{metrics['field_accuracy']:.2%}"
        )

        print(
            f"Exact Match Accuracy:  "
            f"{metrics['exact_match_accuracy']:.2%}"
        )

        print(
            f"Precision:             "
            f"{metrics['precision']:.2%}"
        )

        print(
            f"Recall:                "
            f"{metrics['recall']:.2%}"
        )

        print(
            f"F1-score:              "
            f"{metrics['f1_score']:.2%}"
        )

        print(
            f"JSON Validity:         "
            f"{metrics['json_validity']:.2%}"
        )

        print(
            f"Total Examples:        "
            f"{metrics['total_examples']}"
        )

        print("=" * 55)