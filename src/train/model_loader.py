import json
import torch

from pathlib import Path
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from datasets import Dataset

import os


class ModelManager:
    """
    Manage LLM loading, GPU configuration,
    prompt formatting and dataset preparation.
    """

    def __init__(self, env_path: str = ".env"):

        # Load environment variables
        load_dotenv(dotenv_path=env_path)

        # Model configuration
        self.model_name = os.getenv(
            "MODEL_NAME"
        )

        self.max_seq_length = int(
            os.getenv("MAX_SEQ_LENGTH")
        )

        self.load_in_4bit = (
            os.getenv("LOAD_IN_4BIT")).strip().lower() == "true"

        # Model components
        self.model = None
        self.tokenizer = None

    # --------------------------------------------------
    # 1. GPU CHECK
    # --------------------------------------------------

    def check_gpu(self) -> None:
        """
        Check CUDA availability and display GPU information.
        """

        print("=" * 60)
        print("GPU CHECK")
        print("=" * 60)

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():

            print(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

            print(
                f"CUDA version: {torch.version.cuda}"
            )

            print(
                f"GPU memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

        else:
            print("GPU: None")

        print("=" * 60)

    # --------------------------------------------------
    # 2. MODEL LOADING
    # --------------------------------------------------

    def load_model(self):
        """
        Load the base LLM and its tokenizer using Unsloth.

        Returns:
            model, tokenizer
        """

        self.check_gpu()

        print("\n" + "=" * 60)
        print("LOADING MODEL")
        print("=" * 60)

        print(f"Model: {self.model_name}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"4-bit quantization: {self.load_in_4bit}")

        self.model, self.tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=self.model_name,

                max_seq_length=self.max_seq_length,

                dtype=None,

                load_in_4bit=self.load_in_4bit,
            )
        )

        print("\n✓ Model loaded successfully.")

        return self.model, self.tokenizer

    # --------------------------------------------------
    # 3. PROMPT FORMATTING
    # --------------------------------------------------

    @staticmethod
    def format_prompt(example: dict) -> str:
        """
        Convert one example into the fine-tuning prompt format.

        Input: HTML product information
        Output: Structured JSON product information
        """

        if "input" not in example or "output" not in example:
            raise ValueError(
                "Each example must contain 'input' and 'output'."
            )

        return (
            f"### Input: {example['input']}\n"
            f"### Output: {json.dumps(example['output'], ensure_ascii=False)}"
            f"<|endoftext|>"
        )

    # --------------------------------------------------
    # 4. DATASET PREPARATION
    # --------------------------------------------------

    @classmethod
    def prepare_dataset(cls, data: list[dict]) -> Dataset:
        """
        Convert Python examples into a Hugging Face Dataset.
        """

        if not data:
            raise ValueError("Dataset is empty.")

        formatted_data = [
            cls.format_prompt(example)
            for example in data
        ]

        dataset = Dataset.from_dict({
            "text": formatted_data
        })

        print("\nDataset prepared successfully.")
        print(f"Number of examples: {len(dataset)}")

        return dataset

    # --------------------------------------------------
    # 5. MODEL INFORMATION
    # --------------------------------------------------

    def get_model_info(self) -> dict:
        """
        Return the current model configuration.
        """

        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        }
