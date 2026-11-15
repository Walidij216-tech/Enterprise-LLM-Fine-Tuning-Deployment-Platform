import json
import os

from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load environment variables from the project root
load_dotenv()


class DatasetQualityAnalyzer:
    """Analyze the quality of a structured JSON dataset."""

    REQUIRED_FIELDS = [
        "name",
        "price",
        "category",
        "manufacturer",
    ]

    def __init__(self, dataset_path: str | Path | None = None):
        """
        Initialize the analyzer.

        Args:
            dataset_path: Dataset path. If omitted, DATASET_PATH
                          is read from the .env file.
        """
        configured_path = dataset_path or os.getenv("DATASET_PATH")

        if not configured_path:
            raise ValueError(
                "DATASET_PATH is not configured in the .env file."
            )

        self.dataset_path = Path(configured_path)
        self.dataset: list[dict[str, Any]] = []
        
    def _get_dataset(self, dataset=None) -> list[dict[str, Any]]:
        """
        Return the provided dataset or load it from the configured path.

        Args:
            dataset: Optional dataset already loaded into memory.

        Returns:
            A list of dataset examples.
        """
        # Reuse the provided dataset if available
        if dataset is not None:
            return dataset

        # Reuse the cached dataset if it has already been loaded
        if self.dataset:
            return self.dataset


    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    def count_samples(self, dataset=None) -> int:
        """Return the total number of samples."""
        return len(self._get_dataset(dataset))

    def count_unique_products(self, dataset=None) -> int:
        """Count unique product names."""
        products = set()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            output = sample.get("output", {})
            if not isinstance(output, dict):
                continue

            name = output.get("name")

            if isinstance(name, str) and name.strip():
                products.add(name.strip())

        return len(products)

    def count_categories(self, dataset=None) -> int:
        """Count unique categories."""
        categories = set()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            output = sample.get("output", {})
            if not isinstance(output, dict):
                continue

            category = output.get("category")

            if isinstance(category, str) and category.strip():
                categories.add(category.strip())

        return len(categories)

    def count_manufacturers(self, dataset=None) -> int:
        """Count unique manufacturers."""
        manufacturers = set()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            output = sample.get("output", {})
            if not isinstance(output, dict):
                continue

            manufacturer = output.get("manufacturer")

            if isinstance(manufacturer, str) and manufacturer.strip():
                manufacturers.add(manufacturer.strip())

        return len(manufacturers)

    # ---------------------------------------------------------
    # Distribution analysis
    # ---------------------------------------------------------

    def category_distribution(self, dataset=None) -> Counter:
        """Count examples belonging to each category."""
        counter = Counter()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            output = sample.get("output", {})
            if not isinstance(output, dict):
                continue

            category = output.get("category")

            if isinstance(category, str) and category.strip():
                counter[category.strip()] += 1

        return counter

    def manufacturer_distribution(self, dataset=None) -> Counter:
        """Count examples belonging to each manufacturer."""
        counter = Counter()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            output = sample.get("output", {})
            if not isinstance(output, dict):
                continue

            manufacturer = output.get("manufacturer")

            if isinstance(manufacturer, str) and manufacturer.strip():
                counter[manufacturer.strip()] += 1

        return counter

    # ---------------------------------------------------------
    # Missing fields
    # ---------------------------------------------------------

    def find_missing_fields(self, dataset=None) -> dict[str, int]:
        """Count missing or empty output fields."""
        data = self._get_dataset(dataset)

        missing = {field: 0 for field in self.REQUIRED_FIELDS}

        for sample in data:
            output = (
                sample.get("output", {})
                if isinstance(sample, dict)
                else {}
            )

            if not isinstance(output, dict):
                output = {}

            for field in self.REQUIRED_FIELDS:
                value = output.get(field)

                if value is None or (
                    isinstance(value, str) and not value.strip()
                ):
                    missing[field] += 1

        return missing

    def missing_field_percentages(self, dataset=None) -> dict[str, float]:
        """Calculate missing-value percentages."""
        data = self._get_dataset(dataset)
        total = len(data)
        missing = self.find_missing_fields(data)

        if total == 0:
            return {field: 0.0 for field in self.REQUIRED_FIELDS}

        return {
            field: count / total * 100
            for field, count in missing.items()
        }

    # ---------------------------------------------------------
    # Duplicate analysis
    # ---------------------------------------------------------

    def find_duplicate_inputs(self, dataset=None) -> dict[str, int]:
        """Return input strings occurring more than once."""
        counter = Counter()

        for sample in self._get_dataset(dataset):
            if not isinstance(sample, dict):
                continue

            input_text = sample.get("input")

            if isinstance(input_text, str) and input_text.strip():
                counter[input_text.strip()] += 1

        return {
            text: count
            for text, count in counter.items()
            if count > 1
        }

    def find_duplicate_examples(self, dataset=None) -> int:
        """Count extra occurrences of identical input-output examples."""
        serialized_examples = []

        for sample in self._get_dataset(dataset):
            serialized_examples.append(
                json.dumps(
                    sample,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )

        counter = Counter(serialized_examples)

        return sum(
            count - 1
            for count in counter.values()
            if count > 1
        )

    # ---------------------------------------------------------
    # Input / output length analysis
    # ---------------------------------------------------------

    def analyze_input_lengths(self, dataset) -> dict[str, float]:
        """Analyze input lengths in characters."""
        lengths = []

        for sample in dataset:
            input_text = (
                sample.get("input", "")
                if isinstance(sample, dict)
                else ""
            )
            lengths.append(
                len(input_text) if isinstance(input_text, str) else 0
            )

        if not lengths:
            return {"min": 0, "max": 0, "average": 0.0}

        return {
            "min": min(lengths),
            "max": max(lengths),
            "average": sum(lengths) / len(lengths),
        }

    def analyze_output_lengths(self, dataset=None) -> dict[str, float]:
        """Analyze serialized output lengths in characters."""
        lengths = []

        for sample in self._get_dataset(dataset):
            output = (
                sample.get("output", {})
                if isinstance(sample, dict)
                else {}
            )

            output_json = json.dumps(
                output,
                ensure_ascii=False,
            )
            lengths.append(len(output_json))

        if not lengths:
            return {"min": 0, "max": 0, "average": 0.0}

        return {
            "min": min(lengths),
            "max": max(lengths),
            "average": sum(lengths) / len(lengths),
        }

    # ---------------------------------------------------------
    # Invalid examples
    # ---------------------------------------------------------

    def find_invalid_examples(self, dataset=None) -> list[int]:
        """
        Return zero-based indices of examples with an invalid structure.

        Expected:
        {
            "input": str,
            "output": {
                "name": str,
                "price": str,
                "category": str,
                "manufacturer": str
            }
        }
        """
        invalid_indices = []

        for index, sample in enumerate(self._get_dataset(dataset)):
            if not isinstance(sample, dict):
                invalid_indices.append(index)
                continue

            input_text = sample.get("input")
            output = sample.get("output")

            if not isinstance(input_text, str) or not isinstance(output, dict):
                invalid_indices.append(index)
                continue

            if any(
                field not in output
                or not isinstance(output[field], str)
                or not output[field].strip()
                for field in self.REQUIRED_FIELDS
            ):
                invalid_indices.append(index)

        return invalid_indices

    # ---------------------------------------------------------
    # Report generation
    # ---------------------------------------------------------

    def generate_report(self, data) -> dict[str, Any]:
        """Generate the complete quality report."""


        input_lengths = self.analyze_input_lengths(data)
        output_lengths = self.analyze_output_lengths(data)

        return {
            "samples": self.count_samples(data),
            "unique_products": self.count_unique_products(data),
            "categories": self.count_categories(data),
            "manufacturers": self.count_manufacturers(data),
            "category_distribution": dict(
                self.category_distribution(data)
            ),
            "manufacturer_distribution": dict(
                self.manufacturer_distribution(data)
            ),
            "missing_fields": self.find_missing_fields(data),
            "missing_field_percentages": (
                self.missing_field_percentages(data)
            ),
            "duplicate_inputs": len(
                self.find_duplicate_inputs(data)
            ),
            "duplicate_examples": self.find_duplicate_examples(data),
            "invalid_examples": len(
                self.find_invalid_examples(data)
            ),
            "input_length": input_lengths,
            "output_length": output_lengths,
        }

    # ---------------------------------------------------------
    # Report printing
    # ---------------------------------------------------------

    def print_report(self, report: dict[str, Any]) -> None:
        """Print a formatted quality report."""
        print("\n" + "=" * 60)
        print("DATASET QUALITY REPORT")
        print("=" * 60)

        print(f"Samples:              {report['samples']}")
        print(f"Unique products:      {report['unique_products']}")
        print(f"Categories:           {report['categories']}")
        print(f"Manufacturers:        {report['manufacturers']}")

        print("\n" + "-" * 60)
        print("CATEGORY DISTRIBUTION")
        print("-" * 60)

        total = report["samples"]

        for category, count in sorted(
            report["category_distribution"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            percentage = count / total * 100 if total else 0
            print(f"{category:<20} {count:>5} ({percentage:.2f}%)")

        print("\n" + "-" * 60)
        print("MANUFACTURER DISTRIBUTION")
        print("-" * 60)

        for manufacturer, count in sorted(
            report["manufacturer_distribution"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            percentage = count / total * 100 if total else 0
            print(f"{manufacturer:<20} {count:>5} ({percentage:.2f}%)")

        print("\n" + "-" * 60)
        print("MISSING FIELDS")
        print("-" * 60)

        for field in self.REQUIRED_FIELDS:
            count = report["missing_fields"][field]
            percentage = report["missing_field_percentages"][field]
            print(f"{field:<20} {count:>5} ({percentage:.2f}%)")

        print("\n" + "-" * 60)
        print("DUPLICATES / INVALID DATA")
        print("-" * 60)

        print(f"Duplicate inputs:    {report['duplicate_inputs']}")
        print(f"Duplicate examples:  {report['duplicate_examples']}")
        print(f"Invalid examples:    {report['invalid_examples']}")

        print("\n" + "-" * 60)
        print("INPUT LENGTH")
        print("-" * 60)

        print(f"Min:                  {report['input_length']['min']:.0f}")
        print(f"Max:                  {report['input_length']['max']:.0f}")
        print(f"Average:              {report['input_length']['average']:.2f}")

        print("\n" + "-" * 60)
        print("OUTPUT LENGTH")
        print("-" * 60)

        print(f"Min:                  {report['output_length']['min']:.0f}")
        print(f"Max:                  {report['output_length']['max']:.0f}")
        print(f"Average:              {report['output_length']['average']:.2f}")

        print("=" * 60)





