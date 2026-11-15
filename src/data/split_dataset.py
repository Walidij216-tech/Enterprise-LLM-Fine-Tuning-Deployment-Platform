import json
import re
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import os



class DatasetSplitter:
    """
    Manage dataset quality checks, splitting, leakage detection,
    and saving train/validation/test datasets.
    """

    def __init__(self,output_dir):
        """Initialize the splitter using environment variables."""

        self.output_dir = Path(output_dir)

        self.random_state = int(
            os.getenv("RANDOM_STATE")
        )

        self.train_ratio = float(
            os.getenv("TRAIN_RATIO")
        )

        self.val_ratio = float(
            os.getenv("VAL_RATIO")
        )

        self.test_ratio = float(
            os.getenv("TEST_RATIO")
        )

        self.similarity_threshold = float(
            os.getenv("SIMILARITY_THRESHOLD")
        )



    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace and case for text comparison."""

        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ========================================================
    # EXACT DUPLICATES
    # ========================================================

    def find_exact_duplicates(
        self,
        dataset: list[dict]
    ) -> list[tuple[int, int]]:
        """Find pairs of examples with identical normalized inputs."""

        seen = {}
        duplicates = []

        for index, example in enumerate(dataset):
            input_text = self.normalize_text(example["input"])

            if input_text in seen:
                duplicates.append((seen[input_text], index))
            else:
                seen[input_text] = index

        return duplicates

    def report_exact_duplicates(
        self,
        dataset: list[dict],
        duplicates: list[tuple[int, int]]
    ) -> None:
        """Print exact duplicate examples."""

        print("\n" + "=" * 60)
        print("EXACT DUPLICATES")
        print("=" * 60)

        print(f"Duplicate pairs: {len(duplicates)}")

        for index1, index2 in duplicates:
            print(f"\nExample {index1} <-> Example {index2}")
            print("Input:")
            print(dataset[index1]["input"])
            print("Output 1:")
            print(dataset[index1]["output"])
            print("Output 2:")
            print(dataset[index2]["output"])
            print("-" * 60)

    # ========================================================
    # NEAR DUPLICATES
    # ========================================================

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two normalized inputs."""

        return SequenceMatcher(
            None,
            self.normalize_text(text1),
            self.normalize_text(text2)
        ).ratio()

    def find_near_duplicates(
        self,
        dataset: list[dict]
    ) -> list[tuple[int, int, float]]:
        """
        Find pairs of inputs whose similarity reaches the threshold.

        Note: structurally similar HTML templates can produce many
        matches even when the product information differs.
        """

        near_duplicates = []

        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                score = self.similarity(
                    dataset[i]["input"],
                    dataset[j]["input"]
                )

                if score >= self.similarity_threshold:
                    near_duplicates.append((i, j, score))

        return near_duplicates

    # ========================================================
    # SPLIT DATASET
    # ========================================================

    def split_dataset(
        self,
        dataset: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split the dataset using configured ratios."""

        if len(dataset) < 3:
            raise ValueError(
                "At least 3 examples are needed for three splits."
            )

        train, temporary = train_test_split(
            dataset,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_state,
            shuffle=True
        )

        validation, test = train_test_split(
            temporary,
            test_size=(
                self.test_ratio /
                (self.val_ratio + self.test_ratio)
            ),
            random_state=self.random_state,
            shuffle=True
        )

        return train, validation, test

    # ========================================================
    # DATA LEAKAGE
    # ========================================================

    def check_data_leakage(
        self,
        train: list[dict],
        validation: list[dict],
        test: list[dict]
    ) -> bool:
        """
        Detect exact normalized inputs shared across splits.

        Returns True if any cross-split duplicates are found.
        """

        def build_input_index(dataset):
            index = {}

            for position, example in enumerate(dataset):
                normalized = self.normalize_text(example["input"])
                index.setdefault(normalized, []).append(position)

            return index

        train_inputs = build_input_index(train)
        validation_inputs = build_input_index(validation)
        test_inputs = build_input_index(test)

        train_val = set(train_inputs) & set(validation_inputs)
        train_test = set(train_inputs) & set(test_inputs)
        validation_test = (
            set(validation_inputs) & set(test_inputs)
        )

        print("\n" + "=" * 60)
        print("DATA LEAKAGE CHECK")
        print("=" * 60)

        print(f"Train <-> Validation: {len(train_val)}")
        print(f"Train <-> Test:       {len(train_test)}")
        print(f"Validation <-> Test:  {len(validation_test)}")

        duplicate_groups = [
            ("TRAIN <-> VALIDATION", train_val,
             train_inputs, validation_inputs, train, validation),
            ("TRAIN <-> TEST", train_test,
             train_inputs, test_inputs, train, test),
            ("VALIDATION <-> TEST", validation_test,
             validation_inputs, test_inputs, validation, test),
        ]

        for (
            title,
            common_inputs,
            left_index,
            right_index,
            left_data,
            right_data
        ) in duplicate_groups:

            if not common_inputs:
                continue

            print(f"\n{title} DUPLICATE EXAMPLES")

            for input_text in common_inputs:
                for left_position in left_index[input_text]:
                    for right_position in right_index[input_text]:
                        print(f"\nFirst split index:  {left_position}")
                        print(f"Second split index: {right_position}")
                        print(f"Input: {input_text}")
                        print(
                            "First output:",
                            left_data[left_position]["output"]
                        )
                        print(
                            "Second output:",
                            right_data[right_position]["output"]
                        )

        leakage_found = bool(
            train_val or train_test or validation_test
        )

        if leakage_found:
            print("\nWARNING: Potential data leakage detected.")
        else:
            print("\nNo exact cross-split duplicates found.")

        return leakage_found

    # ========================================================
    # SAVE SPLITS
    # ========================================================

    def save_dataset(
        self,
        dataset: list[dict],
        filename: str
    ) -> Path:
        """Save a split as a JSON file."""

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = self.output_dir / filename

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                dataset,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved: {output_path}")

        return output_path

    # ========================================================
    # SPLIT REPORT
    # ========================================================

    @staticmethod
    def print_split_report(
        full_dataset: list[dict],
        train: list[dict],
        validation: list[dict],
        test: list[dict]
    ) -> None:
        """Display the size of each split."""

        total = len(full_dataset)

        print("\n" + "=" * 60)
        print("TRAIN / VALIDATION / TEST SPLIT")
        print("=" * 60)

        print(f"Total samples:      {total}")
        print(
            f"Train:              {len(train):>3} "
            f"({len(train) / total * 100:.1f}%)"
        )
        print(
            f"Validation:         {len(validation):>3} "
            f"({len(validation) / total * 100:.1f}%)"
        )
        print(
            f"Test:               {len(test):>3} "
            f"({len(test) / total * 100:.1f}%)"
        )

        print("=" * 60)
