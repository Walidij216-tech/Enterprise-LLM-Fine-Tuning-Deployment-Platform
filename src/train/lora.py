import os
from typing import Optional

from dotenv import load_dotenv
from unsloth import FastLanguageModel


# Load environment variables from the project root
load_dotenv()


class LoRAConfig:
    """
    Configure and apply LoRA adapters to a base language model.
    """

    def __init__(self):
        # LoRA hyperparameters
        self.r = int(os.getenv("LORA_R"))

        self.alpha = int(os.getenv("LORA_ALPHA"))

        self.dropout = float(os.getenv("LORA_DROPOUT"))

        # Target modules
        target_modules = os.getenv(
            "LORA_TARGET_MODULES",
        )

        self.target_modules = [
            module.strip()
            for module in target_modules.split(",")
            if module.strip()
        ]

        # Gradient checkpointing
        self.gradient_checkpointing = os.getenv(
            "USE_GRADIENT_CHECKPOINTING"
        )

        # Reproducibility
        self.random_state = int(
            os.getenv("LORA_RANDOM_STATE")
        )

        # Optional LoRA variants
        self.use_rslora = (
            os.getenv("USE_RSLORA", "false").lower() == "true"
        )

        self.loftq_config = None

    def display_config(self) -> None:
        """
        Display the current LoRA configuration.
        """

        print("\n" + "=" * 60)
        print("LoRA CONFIGURATION")
        print("=" * 60)

        print(f"Rank (r):                  {self.r}")
        print(f"Alpha:                     {self.alpha}")
        print(f"Dropout:                   {self.dropout}")
        print(f"Gradient checkpointing:    {self.gradient_checkpointing}")
        print(f"Random state:              {self.random_state}")
        print(f"RSLoRA:                    {self.use_rslora}")
        print(f"LoftQ config:              {self.loftq_config}")

        print("\nTarget modules:")

        for module in self.target_modules:
            print(f"  - {module}")

        print("=" * 60)

    def validate_config(self) -> None:
        """
        Validate LoRA hyperparameters.
        """

        if self.r <= 0:
            raise ValueError("LoRA rank must be greater than zero.")

        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be greater than zero.")

        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be between 0 and 1.")

        if not self.target_modules:
            raise ValueError("At least one target module is required.")

    def apply(self, model):
        """
        Apply LoRA adapters to the base model.

        Args:
            model: Base language model.

        Returns:
            Model with LoRA adapters.
        """

        self.validate_config()
        self.display_config()

        print("\nApplying LoRA adapters...")

        model = FastLanguageModel.get_peft_model(
            model,

            r=self.r,

            target_modules=self.target_modules,

            lora_alpha=self.alpha,

            lora_dropout=self.dropout,

            bias="none",

            use_gradient_checkpointing=self.gradient_checkpointing,

            random_state=self.random_state,

            use_rslora=self.use_rslora,

            loftq_config=self.loftq_config,
        )

        print("\n✓ LoRA adapters applied successfully.")

        return model
