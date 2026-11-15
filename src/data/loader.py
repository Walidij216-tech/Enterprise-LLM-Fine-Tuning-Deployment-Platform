
from pathlib import Path
import json
from typing import Any


class DatasetLoader:
    """
    Load a dataset from a JSON file.
    """

    def __init__(self, path: Path):
        """
        Initialize the dataset loader.

        Args:
            path: Path to the JSON dataset.
        """
        self.path = Path(path)

    def load_dataset(self) -> list[dict[str, Any]]:
        """
        Load the dataset from a JSON file.

        Expected format:
        [
            {
                "input": "...",
                "output": {
                    "name": "...",
                    "price": "...",
                    "category": "...",
                    "manufacturer": "..."
                }
            }
        ]
        """

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}"
            )

        with self.path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError(
                "Dataset must be a JSON list."
            )

        return dataset


