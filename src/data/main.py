
import os
from pathlib import Path
from pydantic import ValidationError
from dotenv import load_dotenv

from src.data.loader import DatasetLoader
from src.data.validator import ProductOutput
from src.data.quality_analysis import DatasetQualityAnalyzer
from src.data.split_dataset import DatasetSplitter


def main():


    # --------------------------------------------------------
    # 1. Load environment variables
    # --------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env")

    dataset_path = os.getenv("DATASET_PATH")
    output_dir = os.getenv("OUTPUT_DIR")

    if not dataset_path:
        raise ValueError(
            "DATASET_PATH is not defined in the .env file."
        )

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------
    loader = DatasetLoader(dataset_path)
    dataset = loader.load_dataset()

    # --------------------------------------------------------
    # 3. Validate dataset
    # --------------------------------------------------------

    try:
      dataset = [
        {
            **sample,#copy all the key-value pairs from the dictionary sample into a new dictionary.
            
            "output": ProductOutput.model_validate(
                sample["output"]
            ).model_dump()
        }
        for sample in dataset
    ]

    except ValidationError as error:
      print("\n[FAILED] Dataset validation failed!")
      print(error)
      raise

    else:
      print("\n[SUCCESS] All dataset outputs have been validated successfully!")


    # --------------------------------------------------------
    # 4. Analyze dataset quality
    # --------------------------------------------------------
    analyzer = DatasetQualityAnalyzer()

    report = analyzer.generate_report(dataset)
    analyzer.print_report(report)

    # --------------------------------------------------------
    # 5. Initialize dataset splitter
    # --------------------------------------------------------
    splitter = DatasetSplitter(output_dir)

    # --------------------------------------------------------
    # 6. Detect exact duplicates
    # --------------------------------------------------------
    exact_duplicates = splitter.find_exact_duplicates(dataset)

    print(f"\nExact duplicate inputs: {len(exact_duplicates)}")

    # --------------------------------------------------------
    # 7. Split dataset
    # --------------------------------------------------------
    train, validation, test = splitter.split_dataset(dataset)

    # --------------------------------------------------------
    # 8. Display split report
    # --------------------------------------------------------
    splitter.print_split_report(
        dataset,
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # 9. Check data leakage
    # --------------------------------------------------------
    splitter.check_data_leakage(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # 10. Save split datasets
    # --------------------------------------------------------
    splitter.save_dataset(train, "train.json")
    splitter.save_dataset(validation, "validation.json")
    splitter.save_dataset(test, "test.json")

    print("\nDataset splitting completed successfully.")

if __name__ == "__main__":
    main()
