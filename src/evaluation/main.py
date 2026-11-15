import os
from pathlib import Path
from dotenv import load_dotenv
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.benchmark import Benchmark
from src.data.loader import DatasetLoader
from src.experiments.mlflow_tracker import MLflowTracker

def main():


    # --------------------------------------------------------
    # Load environment variables
    # --------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env")

    dataset_path = os.getenv("TEST_DATASET_PATH")

    print("Starting evaluation...")

    # Initialize classes
    loader=DatasetLoader(dataset_path)
    metrics_calculator = MetricsCalculator()
    evaluator = ModelEvaluator()
    benchmark = Benchmark()

    mlflow_tracker = MLflowTracker()

    run_name = os.getenv(
        "MLFLOW_EVALUATION_RUN_NAME"
    )

    # Start MLflow evaluation run
    with mlflow_tracker.start_run(run_name):

        mlflow_tracker.set_tags({
            "stage": "evaluation",
            "model_name": getattr(
                evaluator,
                "model_name",
                "product-extractor"
            ),
            "evaluation_type": "test"
        })


    # Load test dataset
    test_dataset = loader.load_dataset()

    # Log evaluation configuration
    mlflow_tracker.log_params({
            "test_dataset": str(dataset_path),
            "test_examples": len(test_dataset),
            "evaluation_type": "test"
        })

    # Generate predictions
    predictions = evaluator.evaluate_dataset(test_dataset)

    # Extract expected outputs
    references = [
        example["output"]
        for example in test_dataset
    ]

    # Calculate metrics
    metrics = metrics_calculator.calculate_metrics(
        predictions,
        references
    )

    # Display results
    benchmark.display_metrics(metrics)

    # Save results
    benchmark.save_metrics(metrics)

    benchmark.save_predictions(
        predictions,
        test_dataset
    )

    # Log metrics to MLflow
    mlflow_tracker.log_metrics(
            metrics,
            prefix="test."
        )

        # Save evaluation report in MLflow
    mlflow_tracker.log_dict(
            metrics,
            "test_metrics.json"
        )

    print("Evaluation completed!")


if __name__ == "__main__":
    main()