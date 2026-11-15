
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class MLflowTracker:

    def __init__(self):
        # Build an absolute SQLite database path
        database_path = PROJECT_ROOT / "mlflow.db"
        tracking_uri = f"sqlite:///{database_path.as_posix()}"
        
        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME")

        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name):
        "démarre une expérimentation."
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def _flatten_dict(data, parent_key=""):
        """Convert nested dictionaries into flat MLflow parameters."""
        "enregistrer des configurations même si elles contiennent plusieurs niveaux de dictionnaires."
        flattened = {}

        if not isinstance(data, dict):
            return flattened

        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else str(key)

            if isinstance(value, dict):
                flattened.update(
                    MLflowTracker._flatten_dict(value, new_key)
                )#this is Recursion: a function calls itself.
            elif value is not None and isinstance(
                value, (str, int, float, bool)
            ):
                flattened[new_key] = str(value)[:500]

        return flattened

    @staticmethod
    def log_params(params):
        "enregistre les hyperparamètres"
        if isinstance(params, dict):
            flattened = MLflowTracker._flatten_dict(params)
            if flattened:
                mlflow.log_params(flattened)#a function provided by the MLflow library:Sends the flattened dictionary to MLflow

    @staticmethod
    def log_metrics(metrics, prefix=""):
        """Log numeric metrics only."""
        if not isinstance(metrics, dict):
            return

        numeric_metrics = {}

        for key, value in metrics.items():
            metric_name = f"{prefix}{key}"

            if isinstance(value, dict):
                MLflowTracker.log_metrics(value, f"{metric_name}.")#recursion to find numeric metrics at every dictionary level
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics[metric_name] = float(value)

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

    @staticmethod
    def log_dict(data, artifact_file):
        "sauvegarde les rapports JSON comme artefacts."
        mlflow.log_dict(data, artifact_file)

    @staticmethod
    def log_artifact(file_path):
        "it is used to save a file (such as a JSON report, CSV file, "
        "or configuration file) inside the current MLflow experiment run."
        path = Path(file_path)

        if path.is_file():
            mlflow.log_artifact(str(path))
        else:
            print(f"Artifact not found: {path}")

    @staticmethod
    def set_tags(tags):
        "ajoute des informations comme le type de modèle ou le dataset utilisé."
        mlflow.set_tags(tags)
