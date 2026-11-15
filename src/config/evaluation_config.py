import os
from pathlib import Path
from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load project .env
load_dotenv(PROJECT_ROOT / ".env")


class EvaluationConfig:

    REQUIRED_FIELDS = [
        field.strip()
        for field in os.getenv(
            "REQUIRED_FIELDS",
            "name,price,category,manufacturer"
        ).split(",")
        if field.strip()
    ]

    MODEL_NAME = os.getenv(
        "EVALUATION_MODEL",
        "product-extractor"
    )

    API_URL = os.getenv(
        "EVALUATION_API_URL",
        "http://localhost:11434"
    )

    TEMPERATURE = float(
        os.getenv("EVALUATION_TEMPERATURE", "0.2")
    )

    OUTPUT_DIR = PROJECT_ROOT / os.getenv(
        "EVALUATION_OUTPUT_DIR",
        "evaluation_results"
    )

    TEST_DATASET_PATH = PROJECT_ROOT / os.getenv(
        "TEST_DATASET_PATH",
        "src/data/split/test.json"
    )